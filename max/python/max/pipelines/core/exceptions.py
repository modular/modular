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

"""Defines pipeline-specific exceptions for input validation errors."""

from __future__ import annotations


class InputError(ValueError):
    """Exception raised for input validation errors that should be shown to users.

    This exception is specifically designed to provide user-friendly error messages
    for invalid input data, such as missing images for vision models or incorrect
    parameter combinations.
    """

    pass
