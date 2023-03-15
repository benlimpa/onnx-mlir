#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConvOpt.h"

using namespace mlir;

namespace {

LogicalResult matchAddOpTypes(ONNXAddOp op) {
  if (op.getOperand(0).getType().dyn_cast<TensorType>().getElementType().isa<FloatType>()) {
    if (op.getOperand(0).getType().dyn_cast<TensorType>().getElementType().isa<FloatType>()) {
      return success();
    }
  }
  return failure();
}

struct AddToMulPattern : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult match(ONNXAddOp op) const override {
    return matchAddOpTypes(op);
  }

  void rewrite(ONNXAddOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ONNXMulOp>(
        op, op.getOperand(0), op.getOperand(1));
  }
};

struct ToyPass : public PassWrapper<ToyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyPass)
  ::llvm::StringRef getArgument() const override { return "toy-pass"; }

  ::llvm::StringRef getDescription() const override {
    return "Toy pass to test pattern targetted mutators versus generic "
           "mutators.";
  }

  void runOnOperation() final;
};
void ToyPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<AddToMulPattern>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createToyPass() {
  return std::make_unique<ToyPass>();
}

} // namespace onnx_mlir