#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include <src/libfuzzer/libfuzzer_macro.h>
#include <iostream>
#include <regex>

using namespace onnx_mlir;

DEFINE_PROTO_FUZZER(const onnx::ModelProto &input) {
  mlir::MLIRContext context;
  registerDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  onnx::ModelProto model(input);
  int rc = processInputMsg(model, context, module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    return;
  }

  compileModule(module, context, "onnx-mlir-output", EmitLib);
}