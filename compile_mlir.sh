#!/bin/bash

set -e

cd llvm-project/build

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
    echo >&2 "Usage: ./compile_mlir.sh configure/build/test"
fi


