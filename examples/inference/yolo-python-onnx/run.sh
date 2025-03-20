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

# If anything goes wrong, stop running the script.
set -e

MODEL_DIR="../../models/yolo"

# Make sure we're running from inside the directory containing this file.
cd "$(dirname "$0")"

# If CONDA_PREFIX is set, install requirements
if [[ -n "$CONDA_PREFIX" ]]; then
    python3 -m pip install -r requirements.txt
fi

# Download model
python3 download-model.py -o "$MODEL_DIR"

# Execute model
python3 segment.py
