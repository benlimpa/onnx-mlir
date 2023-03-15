#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace toy_mutators {
struct TriggerAddToMulPattern : public mlir::RewritePattern {
  TriggerAddToMulPattern(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  // mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
  // mlir::PatternRewriter &rewriter) const override;
  mlir::LogicalResult match(mlir::Operation *op) const override;
  void rewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override;
};
} // namespace toy_mutators