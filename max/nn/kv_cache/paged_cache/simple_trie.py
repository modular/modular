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

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from max.profiler import traced

Key = Any


class SimpleNode:
    """A node in a simple trie data structure."""

    def __init__(self) -> None:
        # Pointer to the children of this node
        self.children: dict[Key, SimpleNode] = {}
        # Whether this node is the end of a word
        self.is_eow: bool = False


class SimpleTrie:
    """A simple prefix trie with basic set insertion and deletion operations.

    Unlike the RadixTrie, this trie only have one item at each node and does not
    map multiple keys to multiple values. Instead, it just stores a set of keys.
    """

    def __init__(self) -> None:
        self.root = SimpleNode()

    @traced
    def insert(self, s: Sequence[Key] | npt.NDArray[np.integer[Any]]) -> None:
        """Inserts a sequence into the trie."""
        curr = self.root
        for ch in s:
            if ch not in curr.children:
                curr.children[ch] = SimpleNode()
            curr = curr.children[ch]
        curr.is_eow = True

    @traced
    def _search(
        self, s: Sequence[Key] | npt.NDArray[np.integer[Any]]
    ) -> tuple[SimpleNode, list[Key]]:
        """Internal helper method to search for a sequence in the trie.

        Args:
            s: The sequence to search for

        Returns:
            A tuple containing:
            - The last node reached during the search
            - List of keys matched during the search
        """
        curr = self.root
        matched: list[Key] = []
        for ch in s:
            if ch not in curr.children:
                return curr, matched
            matched.append(ch)
            curr = curr.children[ch]
        return curr, matched

    @traced
    def find_string_with_largest_common_prefix(
        self, target: Sequence[Key] | npt.NDArray[np.integer[Any]]
    ) -> Optional[tuple[Sequence[Key], int]]:
        """Returns a sequence in the trie that shares the longest common prefix
        with the target.

        If a match is found, returns a tuple containing:
        - The matched sequence
        - Length of the common prefix
        Returns None if no match is found
        """
        curr, matched = self._search(target)
        prefix_len = len(matched)

        if prefix_len == 0:
            return None

        # Keep going down left spine to find any string in trie that has s
        # as its prefix
        s = matched
        while not curr.is_eow:
            if not curr.children:
                return None
            ch = next(iter(curr.children))
            curr = curr.children[ch]
            s.append(ch)

        return s, prefix_len

    def __contains__(
        self, s: Sequence[Key] | npt.NDArray[np.integer[Any]]
    ) -> bool:
        """Checks if the trie contains the exact sequence."""
        node, matched = self._search(s)
        return len(matched) == len(s) and node.is_eow

    def __delitem__(
        self, s: Sequence[Key] | npt.NDArray[np.integer[Any]]
    ) -> None:
        """Deletes a sequence from the trie."""

        def _remove(
            root: Optional[SimpleNode], idx: int
        ) -> Optional[SimpleNode]:
            if not root:
                return None

            if idx == len(s):
                assert root.is_eow
                root.is_eow = False
                if len(root.children) == 0:
                    del root
                    root = None
                return root

            ch = s[idx]
            child = _remove(root.children[ch], idx + 1)
            if child is None:
                del root.children[ch]
            else:
                root.children[ch] = child

            if len(root.children) == 0 and not root.is_eow:
                del root
                root = None
            return root

        _remove(self.root, 0)

    def pretty_format(self) -> list[str]:
        """Returns a formatted string representation of the trie for debugging."""
        lines: list[str] = []

        def helper(node: SimpleNode, indent: int) -> None:
            if node.is_eow:
                lines.append(f"{'-' * indent}*")
            for ch, child in node.children.items():
                lines.append(f"{'-' * indent}{ch}")
                helper(child, indent + 2)

        helper(self.root, 0)
        return lines

    def pretty_print(self) -> None:
        """Prints a formatted visual representation of the trie for debugging."""
        print("SimpleTrie(")
        for line in self.pretty_format():
            print(line)
        print(")")
