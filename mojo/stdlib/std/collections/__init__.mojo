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
"""Core data structures: List, Dict, Set, Optional, plus specialized collections.

The `collections` package provides fundamental data structures for organizing
and managing data in Mojo programs. It includes general-purpose collections
like dynamic arrays, hash maps, and sets, along with specialized structures for
specific use cases. These types form the foundation for most data manipulation
in Mojo and integrate seamlessly with the language's iteration and memory
management systems.

Use this package for general data storage and manipulation, implementing
algorithms, building custom data structures, or any task requiring collection
types beyond basic primitives.
"""

from .bitset import BitSet
from .counter import Counter
from .deque import Deque
from .dict import Dict, KeyElement, OwnedKwargsDict
from .inline_array import InlineArray
from .linked_list import LinkedList
from .list import List
from .optional import Optional, OptionalReg
from .set import Set
