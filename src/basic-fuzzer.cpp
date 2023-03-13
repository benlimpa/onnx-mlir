#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include <iostream>
#include <regex>

using namespace onnx_mlir;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // Do basic input validation to avoid stopping early
  if (Size < 1) {
    return 1;
  }

  mlir::MLIRContext context;
  registerDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = processInputArray(Data, Size, context, module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    return 1;
  }

  return compileModule(module, context, "onnx-mlir-output", EmitLib);
}