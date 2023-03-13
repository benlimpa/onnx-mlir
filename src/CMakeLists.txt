# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Accelerators)
add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tools)
add_subdirectory(Version)
add_subdirectory(Compiler)

add_onnx_mlir_executable(onnx-mlir
  onnx-mlir.cpp

  LINK_LIBS PRIVATE
  OMCompilerOptions
  OMCompilerUtils
  )
  
add_onnx_mlir_executable(onnx-mlir-basic-fuzzer
  basic-fuzzer.cpp

  LINK_LIBS PRIVATE
  -fsanitize=fuzzer,address
  OMCompilerOptions
  OMCompilerUtils
)
target_compile_options(onnx-mlir-basic-fuzzer
  PRIVATE -g -O1 -fsanitize=fuzzer,address
)

add_onnx_mlir_executable(onnx-mlir-protobuf-fuzzer
  protobuf-fuzzer.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/third_party/libprotobuf-mutator

  LINK_LIBS PRIVATE
  -fsanitize=fuzzer,address
  OMCompilerOptions
  OMCompilerUtils
  ${ONNX_MLIR_SRC_ROOT}/third_party/libprotobuf-mutator/build/src/libfuzzer/libprotobuf-mutator-libfuzzer.a
  ${ONNX_MLIR_SRC_ROOT}/third_party/libprotobuf-mutator/build/src/libprotobuf-mutator.a
)
target_compile_options(onnx-mlir-protobuf-fuzzer
  PRIVATE -g -O1 -fsanitize=fuzzer,address
)
