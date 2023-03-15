#include <random>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConvOpt.h"
#include "src/Transform/ONNX/ToyMutators.h"

using namespace mlir;

namespace {
struct RandomMutatePass
    : public PassWrapper<RandomMutatePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RandomMutatePass)
    ::llvm::StringRef getArgument() const override { return "mutate-random"; }

    ::llvm::StringRef getDescription() const override {
        return "Randomly select operations in ONNX IR mutatte to trigger various transformations.";
    }

    void runOnOperation() final;
};
void RandomMutatePass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<ONNXDialect, BuiltinDialect, arith::ArithDialect, func::FuncDialect>();

    // TODO get list of operations in the input file
    target.addDynamicallyLegalOp<ONNXDivOp>([&](ONNXDivOp op) {
        // TODO return false with some probability
        return false;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<toy_mutators::TriggerAddToMulPattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

struct WalkerMutatePass
    : public PassWrapper<WalkerMutatePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WalkerMutatePass)
    void runOnOperation() override {
        // TODO choose from a set of multiple mutator patterns
        //RewritePatternSet patterns(&getContext());
        //patterns.add<toy_mutators::TriggerAddToMulPattern>(&getContext());
        toy_mutators::TriggerAddToMulPattern pattern(&getContext());
        getOperation()->walk([&](mlir::Operation *op) {
            
            // get the set of available mutators for this operation
            // TODO: for pattern in patterns:
            if (succeeded(pattern.match(op))) {
                // matchedPatterns.add(pattern)
            }

            // TODO: choose one of the patterns based on a heuristic
            // TODO: pattern = choosePattern(matchedPatterns)

            // apply the chosen pattern
            OpBuilder opBuilder(op);
            PatternRewriter rewriter(getContext());
            pattern.rewrite(op, opBuilder);
        });
    }
};
}

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createRandomMutatePass() {
    return std::make_unique<RandomMutatePass>();
}
std::unique_ptr<mlir::Pass> createWalkerMutatePass() {
    return std::make_unique<WalkerMutatePass>();
}

}