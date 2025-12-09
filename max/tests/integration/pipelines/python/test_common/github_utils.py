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
"""GitHub Actions utilities."""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def github_log_group(title: str) -> Generator[None, None, None]:
    """
    Context manager that creates a collapsible log group in GitHub Actions.

    The group is collapsed by default, hiding the enclosed output unless
    the user expands it in the GitHub Actions log viewer.

    Args:
        title: The title of the group.
    """
    print(f"::group::{title}", flush=True)
    try:
        yield
    finally:
        print("::endgroup::", flush=True)
