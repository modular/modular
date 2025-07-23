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

"""
A custom llvm-lit executor that remaps absolute bazel paths to relative paths
that can be executed outside of bazel.
"""

import re

import lit
from lit.formats import ShTest

_MODULAR_PATH_REGEX = re.compile(r"(^| |=)/[^ ]+/_main/", flags=re.MULTILINE)
_EXTERNAL_PATH_REGEX = re.compile(
    r"(^| |=)/[^ ]+/external/", flags=re.MULTILINE
)


class ModularShTest(ShTest):
    def execute(self, test, litConfig):  # noqa: ANN001
        result = lit.TestRunner.executeShTest(
            test,
            litConfig,
            self.execute_external,
        )

        new_output = _MODULAR_PATH_REGEX.sub(r"\1", result.output)
        new_output = _EXTERNAL_PATH_REGEX.sub(r"\1/external/", new_output)
        result.output = new_output
        return result
