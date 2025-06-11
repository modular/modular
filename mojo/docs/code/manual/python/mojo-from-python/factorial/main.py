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

# get directory of current file
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

# Install mojo import hook
import max.mojo.importer  # noqa: F401
import mojo_module  # type: ignore

print(mojo_module.factorial(5))
