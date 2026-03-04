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
"""Shared GPU kernel primitives for structured kernel architectures.

This package provides architecture-agnostic building blocks used by SM90,
SM100, and other GPU kernel implementations:

- pipeline: Producer-consumer pipeline synchronization
- pipeline_storage: Barrier pair storage and pipeline factory
- tile_types: TileTensor-based shared memory tile abstractions
- kernel_common: Warp role dispatch and kernel context
- barriers: Composable barrier storage for SMEM structs
"""
