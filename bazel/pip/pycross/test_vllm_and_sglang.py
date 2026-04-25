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

# Verifies that `vllm` and `sglang` are importable exactly when enabled

import os


def test_vllm() -> None:
    if os.getenv("USE_VLLM") == "1":
        import vllm  # type: ignore

        assert vllm.__version__


def test_sglang() -> None:
    if os.getenv("USE_SGLANG") == "1":
        import sglang  # type: ignore

        assert sglang.__version__
