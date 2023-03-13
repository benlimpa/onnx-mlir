#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "${BASH_SOURCE}")

# MLIR_DIR must be set with cmake option now
MLIR_DIR=$SCRIPT_DIR/llvm-project/build/lib/cmake/mlir

#pythonLocation=/home/blimpan/mambaforge/envs/onnx

cd $SCRIPT_DIR/build

if [[ $1 == "configure" ]]
then
#            -DPython3_ROOT_DIR=$pythonLocation \
    cmake -G Ninja \
            -DMLIR_DIR=${MLIR_DIR} \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DCMAKE_BUILD_TYPE=Debug \
            ..
elif [[ $1 == "build" ]]
then
    cmake --build .
elif [[ $1 == "test" ]]
then
    # Run lit tests:
    export LIT_OPTS=-v
    cmake --build . --target check-onnx-lit
else
    echo >&2 "Usage: ./compile_onnxmlir.sh configure/build/test"
fi

