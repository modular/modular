# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

cwd = os.path.dirname(__file__)
sys.path.append(cwd)
import cuda_gdb_runner

# name: The name of this test suite.
config.name = "mojo-debug-gpu"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join("Mojo", "test", "max_license", "mojo-debug-gpu")


# The following enables cuda-gdb tests. It doesn't affect tests that don't use
# cuda-gdb.
config.test_format = cuda_gdb_runner.CudaGDBTest(config=config)

# Handle the very rare case of cuda-gdb getting incomplete debug-info from cuda
# driver.
config.test_retry_attempts = 3

config.available_features.add("debugging-test")
