#!/bin/bash

set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

cd $SCRIPT_DIR/third_party/libprotobuf-mutator
mkdir build
cd build
cmake .. -GNinja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
ninja check
