# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys

# Put the current directory (containing module.so) on the Python module lookup
# path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module

if __name__ == "__main__":
    result = mojo_module.mojo_count_args(1, 2)

    assert result == 2

    print("Result from Mojo 🔥:", result)
