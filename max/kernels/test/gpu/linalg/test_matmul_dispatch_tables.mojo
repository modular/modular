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

from linalg.matmul.gpu.sm90.dispatch import (
    llama_8b_fp8_table,
    llama_405b_fp8_table,
)


def main():
    __comptime_assert llama_8b_fp8_table.check()
    __comptime_assert llama_405b_fp8_table.check()
