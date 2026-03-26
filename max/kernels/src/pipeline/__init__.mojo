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
"""Generic compile-time software pipeline scheduling framework.

Import from submodules directly:

    from pipeline.types import OpDesc, ResourceKind, OpRole
    from pipeline.config import PipelineConfig, ScheduleConfig
    from pipeline.compiler import PipelineSchedule, compile_schedule
    from pipeline.schedulers import optimal_schedule_with_halves
    from pipeline.program_builder import verify_schedule, build_kernel_program
"""
