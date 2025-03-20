#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

set -e

CURRENT_DIR=$(dirname "$0")
# Make sure we're running from inside the directory containing this file.
cd "$CURRENT_DIR"

# remove the build artifacts
rm -rf build

if [[ -n "$CONDA_PREFIX" ]]; then
  # Required for the CMake build
  MAX_PKG_DIR="${MAX_PKG_DIR:-$CONDA_PREFIX}"
else
  echo "This script should be run with Conda. Try \`magic run run.sh\`." && false
fi

export MAX_PKG_DIR

MODEL_PATH="$CURRENT_DIR/../../models/bert.torchscript"

# Example input for the model
INPUT_EXAMPLE="My dog is cute."

# Download model from HuggingFace
python3 "$CURRENT_DIR/../common/bert-torchscript/download-model.py" -o "$MODEL_PATH"
python3 "$CURRENT_DIR/pre-process.py" --text "$INPUT_EXAMPLE"

# Build the example
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Run example
./build/bert "$MODEL_PATH"

# Post process
python3 "$CURRENT_DIR/post-process.py"
