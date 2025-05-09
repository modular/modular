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
"""
Provides information about MAX Engine, such as the version.
"""
from ._engine_impl import _EngineImpl, _get_engine_path


fn get_version() raises -> String:
    """Returns the current MAX Engine version.

    Returns:
        Version as string.
    """
    var version = _EngineImpl(_get_engine_path()).get_version()
    return version
