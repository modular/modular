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
"""Kernels whose registrations are wrong, for the load-time errors."""

import extensibility

from max.gpu.host import DeviceContext

from extensibility import InputTensor, OutputTensor


@extensibility.register("context_not_last")
struct ContextNotLast:
    @staticmethod
    def execute(
        output: OutputTensor,
        ctx: DeviceContext,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
    ):
        pass


@extensibility.register("output_after_input")
struct OutputAfterInput:
    @staticmethod
    def execute(
        x: InputTensor,
        output: OutputTensor[dtype=x.dtype, rank=x.rank, ...],
    ):
        pass
