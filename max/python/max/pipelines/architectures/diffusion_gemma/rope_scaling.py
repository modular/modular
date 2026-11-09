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
# The proportional RoPE scaling dataclass, kept on its own so the config and
# the rotary embedding can read it without the layer module.
"""Proportional RoPE scaling parameters, shared by the config and layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProportionalScalingParams:
    """Scaling parameters for proportional RoPE frequency scaling."""

    partial_rotary_factor: float
    """If less than 1.0, inverse frequencies will be returned for the first
    fraction of the head_dim. Defaults to 1.0."""
