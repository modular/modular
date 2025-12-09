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

from collections import deque

import max.nn as nn


def all_subclasses() -> set[type[nn.Module]]:
    """Recursively collects all subclasses of nn.Module."""
    all_subclasses = set()

    q = deque([nn.Module])

    while q:
        cls = q.popleft()
        if cls not in all_subclasses:
            all_subclasses.add(cls)
            q.extend(cls.__subclasses__())

    # Exclude nn.Module from the set.
    all_subclasses.remove(nn.Module)

    return all_subclasses
