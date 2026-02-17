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
"""Smoke test for Mamba architecture: verify module and model load."""

from __future__ import annotations


def test_mamba_arch_imports() -> None:
    """Mamba architecture and model can be imported."""
    from max.pipelines.architectures.mamba import mamba_arch

    assert mamba_arch.name == "MambaForCausalLM"
    assert mamba_arch.pipeline_model is not None
