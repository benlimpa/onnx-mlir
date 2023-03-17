#include <random>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ToyMutators.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

namespace {
struct RandomMutatePass
    : public PassWrapper<RandomMutatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RandomMutatePass)
  ::llvm::StringRef getArgument() const override { return "mutate-random"; }

  ::llvm::StringRef getDescription() const override {
    return "Randomly select operations in ONNX IR mutatte to trigger various "
           "transformations.";
  }

  void runOnOperation() final;
};
void RandomMutatePass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<ONNXDialect, BuiltinDialect, arith::ArithDialect,
      func::FuncDialect>();

  // TODO get list of operations in the input file
  target.addDynamicallyLegalOp<ONNXDivOp>([&](ONNXDivOp op) {
    // TODO return false with some probability
    return false;
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<toy_mutators::TriggerAddToMulPattern>(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

const RewritePattern *choosePattern(
    llvm::SmallVector<const RewritePattern *> patterns) {
  return patterns[0];
}

class MutationPatternRewriteDriver : public PatternRewriter {
public:
  explicit MutationPatternRewriteDriver(
      MLIRContext *ctx, const FrozenRewritePatternSet &patterns, Operation *op)
      : PatternRewriter(ctx), patterns(patterns), matcher(patterns), op(op),
        context(ctx) {}

  LogicalResult mutate() {
    op->walk([&](mlir::Operation *op) {
      // get the set of available mutators for this operation
      llvm::SmallVector<const RewritePattern *> matchedPatterns;
      for (const auto &it : patterns.getOpSpecificNativePatterns()) {
        if (op->getName() == it.first) {
          for (const RewritePattern *pattern : it.second) {
            if (succeeded(pattern->match(op))) {
              matchedPatterns.push_back(pattern);
            }
          }
        }
      }
      for (const auto &pattern : patterns.getMatchAnyOpNativePatterns()) {
        if (succeeded(pattern.match(op))) {
          matchedPatterns.push_back(&pattern);
        }
      }
      auto pattern = choosePattern(matchedPatterns);
      pattern->rewrite(op, *this);
    });
  }

private:
  const FrozenRewritePatternSet &patterns;
  PatternApplicator matcher;
  Operation *op;
  MLIRContext *context;
};
struct WalkerMutatePass
    : public PassWrapper<WalkerMutatePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WalkerMutatePass)
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<toy_mutators::TriggerAddToMulPattern>(context);

    MutationPatternRewriteDriver driver(
        context, std::move(patterns), getOperation());
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createRandomMutatePass() {
  return std::make_unique<RandomMutatePass>();
}
std::unique_ptr<mlir::Pass> createWalkerMutatePass() {
  return std::make_unique<WalkerMutatePass>();
}

} // namespace onnx_mlir
