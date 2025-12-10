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

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from max import mlir
from max.graph import KernelLibrary
from max.torch import CustomOpLibrary


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture(scope="module")
def kernel_library() -> Generator[KernelLibrary]:
    """Set up the kernel library for the current system."""
    path = Path(os.environ["MODULAR_PYTORCH_CUSTOM_OPS"])
    yield KernelLibrary(mlir.Context(), [path])


# Reset op cache between test functions
@pytest.fixture(scope="function")
def op_library(kernel_library: KernelLibrary) -> Generator[CustomOpLibrary]:
    """Set up the kernel library for the current system."""
    yield CustomOpLibrary(kernel_library)
