#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")

if [[ $1 == "clone" ]]
then
    cd "$SCRIPT_DIR"
    git clone -n https://github.com/llvm/llvm-project.git
    # Check out a specific branch that is known to work with ONNX-MLIR.
    cd llvm-project && git checkout 21f4b84c456b471cc52016cf360e14d45f7f2960 && cd ..
    exit 0
fi
cd "$SCRIPT_DIR/llvm-project"
mkdir build || true
cd build
if [[ $1 == "configure" ]]
then
    cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_TARGETS_TO_BUILD="host" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DLLVM_ENABLE_RTTI=ON \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
elif [[ $1 == "build" ]]
then
    cmake --build .
elif [[ $1 == "test" ]]
then
    cmake --build . --target check-mlir
else
    echo >&2 "Usage: ./compile_mlir.sh clone/configure/build/test"
fi


