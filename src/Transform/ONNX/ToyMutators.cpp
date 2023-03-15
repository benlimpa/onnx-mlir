#include "src/Transform/ONNX/ToyMutators.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace toy_mutators {
mlir::LogicalResult TriggerAddToMulPattern::match(mlir::Operation *op) const {
  /*
   * check if the current operation can be mutated
   */
  // the operation must have two inputs and one output
  if (op->hasTrait<mlir::OpTrait::ZeroRegions>() &&
      op->hasTrait<mlir::OpTrait::OneResult>() &&
      op->hasTrait<mlir::OpTrait::NOperands<2>::Impl>())
    return mlir::success();
  else
    return mlir::failure();
}
void TriggerAddToMulPattern::rewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
      rewriter, op->getLoc());

  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);

  if (!lhs.getType()
           .dyn_cast<mlir::TensorType>()
           .getElementType()
           .isa<mlir::FloatType>())
    lhs = create.onnx.cast(lhs, mlir::TypeAttr::get(rewriter.getF32Type()));
  if (!rhs.getType()
           .dyn_cast<mlir::TensorType>()
           .getElementType()
           .isa<mlir::FloatType>())
    rhs = create.onnx.cast(lhs, mlir::TypeAttr::get(rewriter.getF32Type()));

  rewriter.replaceOpWithNewOp<mlir::ONNXAddOp>(op, lhs, rhs);
}
} // namespace toy_mutators