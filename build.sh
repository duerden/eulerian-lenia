#!/bin/bash
# Build lenia WASM module
# Usage: ./build.sh
# Requires: Emscripten SDK activated (source emsdk_env.sh)

set -e

echo "Building lenia.wasm..."
make clean
make
echo "Done. Output: lenia.js + lenia.wasm"
