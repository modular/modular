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

import os
import sys

# The Mojo importer module will handle compilation of the Mojo files.
import max.mojo.importer  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importing our Mojo module, defined in the `hello_mojo.mojo` file.
import hello_mojo  # type: ignore

if __name__ == "__main__":
    # Calling into a Mojo `passthrough` function from Python:
    result = hello_mojo.passthrough("Hello")
    print(result)
