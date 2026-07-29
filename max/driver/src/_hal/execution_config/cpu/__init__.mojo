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
"""Execution Configurations for CPUs

For now, CPUs are treated as a single target, but this may change in the future.
"""
from _hal.execution_config import ExecutionConfig
from .cpu_execution_config import CPUExecutionConfiguration
from std.collections.type_dict import TypeDict

comptime _ExecutionConfigDictForTarget = TypeDict[
    T=StaticString,
    Trait=ExecutionConfig,
    [
        "cpu:amd64",
        "cpu:arm64",
    ],
    CPUExecutionConfiguration,
    CPUExecutionConfiguration,
]

comptime ExecutionConfigForTarget[
    target: StaticString
] = _ExecutionConfigDictForTarget.get[target]
