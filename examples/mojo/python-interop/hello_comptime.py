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

# add current directory to Path to enable importing Mojo modules from this directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# The Mojo importer module will handle compilation of the Mojo files.
import mojo.importer  # noqa: F401
from mojo.importer import set_comptime_variables

# Set compile-time variables before importing the Mojo module.
# Variables are specified as key-value pairs where both keys and values are strings.
# These variables will be available during Mojo compilation via env_get_string.
set_comptime_variables(
    module_name="hello_comptime_mojo",
    values={"MOJO_COMPTIME_KEY": "world"}
)

# Importing our Mojo module, defined in the `hello_comptime_mojo.mojo` file.
import hello_comptime_mojo  # type: ignore

if __name__ == "__main__":
    # Calling into a Mojo function that uses a compile-time variable:
    result = hello_comptime_mojo.hello()
    print(f"{result=}")
