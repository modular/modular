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

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from max.profiler import traced

from .simple_trie import SimpleTrie

TokenId = Any
BlockId = Any
SeqId = int


from collections import OrderedDict


def align_down(x: int, y: int) -> int:
    return (x // y) * y


def _token_prefix_match_len(
    tokens0: np.ndarray, tokens1: np.ndarray, page_size: int
) -> int:
    """Computes the length of maximum shared prefix of two tokens, aligned by
    `page_size`.

    e.g: _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "cats"], page_size = 1) => 2
         _token_prefix_match_len(["i", "like", "dogs"], ["we", "like", "cats"], page_size = 1) => 0
         _token_prefix_match_len(["i", "like", "dogs"], ["i", "like", "dogs", "and", "cats"], page_size = 1) => 3
    """
    assert len(tokens0) % page_size == 0
    assert len(tokens1) % page_size == 0
    shorter_len = min(len(tokens0), len(tokens1))
    diff = tokens0[:shorter_len] != tokens1[:shorter_len]
    idx = np.nonzero(diff)[0]
    if len(idx) == 0:
        return shorter_len
    return align_down(idx[0], page_size)


def _token_to_key(tokens: np.ndarray, page_size: int) -> tuple[TokenId, ...]:
    assert len(tokens) >= page_size, (
        f"tokens must be at least page_size ({page_size}) long but is only {len(tokens)} tokens"
    )
    return tuple(tokens[:page_size])


class TrieNode:
    """A TrieNode consists of a list of tokens and blocks.

    - Tokens are the ids of the tokens in the sequence.
    - Blocks are the offsets into the KVCache region that back the KV entries
      for a given token. I.e: the page index
    """

    node_id_counter = 0

    def __init__(self) -> None:
        """Constructs a TrieNode."""
        # each node is assigned a unique id to look it up in the lru cache
        self.node_id = TrieNode.node_id_counter
        TrieNode.node_id_counter += 1

        self.children: Dict[tuple[TokenId, ...], TrieNode] = {}
        # Typically in a map, we would have keys mapping to values.
        # To avoid collision with KV cache terminology, we call them tokens and blocks.
        #
        # Only the root should have empty tokens/blocks
        self.tokens: np.ndarray = np.array([])
        self.blocks: List[BlockId] = []
        # Only the root should have a null parent
        self.parent: Optional[TrieNode] = None
        # Sequences that are using the blocks owned by this trie node
        # The node can only be evicted if self.active_seqs is empty
        self.active_seqs: Set[SeqId] = set()
        # A trie containing only the keys in the self.children dict where each
        # key is length exactly page_size
        self.key_trie = SimpleTrie()

    def is_leaf(self) -> bool:
        """Returns true if the node is a leaf node."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Returns true if the node is the root node."""
        return self.parent is None

    def is_evictable(self) -> bool:
        """Returns true if the node is evictable."""
        return not self.is_root() and len(self.active_seqs) == 0

    def find_block_with_largest_common_prefix(
        self, target: Sequence[TokenId]
    ) -> Optional[Tuple[BlockId, int]]:
        """Returns any block in the trie that has the given prefix."""
        res = self.key_trie.find_string_with_largest_common_prefix(target)
        if res is None:
            return None
        key, prefix_len = res
        if prefix_len == 0:
            return None
        assert prefix_len <= len(target)
        return self.children[tuple(key)].blocks[0], prefix_len

    def get_prefix_tokens_and_blocks(self) -> Tuple[np.ndarray, List[BlockId]]:
        curr: Optional[TrieNode] = self
        tokens: List[TokenId] = []
        blocks: List[BlockId] = []
        while curr is not None:
            tokens.extend(curr.tokens[::-1])
            blocks.extend(curr.blocks[::-1])
            curr = curr.parent
        return np.array(tokens, dtype=np.int64)[::-1], blocks[::-1]


class LRUCache(OrderedDict):
    """Least recently used block cache to support O(1) eviction operations."""

    def __init__(self):
        super().__init__()

    def __setitem__(self, key: int, value: TrieNode) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

    def move_to_front(self, key: int) -> None:
        super().move_to_end(key, last=False)

    def pop_front(self) -> TrieNode:
        return super().popitem(last=False)[-1]


class RadixTrie:
    """This RadixTrie is specially designed for prefix caching in paged attention.

    The RadixTrie allows for efficient insertion and matching of sequences. It
    matches each prefix of tokens in a sequence to its corresponding blocks.
    Compared to a naive trie, the RadixTrie allows storing multiple tokens in a
    single node for less indirection and faster access.

    Blocks in the RadixTrie should be immutable and committed. If it is in the
    RadixTrie, it is eligible for sharing. An inflight or uncommitted block that
    is being written to by a sequence should not be in the RadixTrie.

    The RadixTrie allows for an LRU eviction policy for its leaves. We only allow
    evictions if no active sequences are using the node.

    Currently, the RadixTrie assumes that the paged KVCache page size is 1.

    This implementation is based off of SGLang:
        - https://github.com/sgl-project/sglang/blob/337fe53ac41c68d6f171ef3b446f55eb0e98f77c/python/sglang/srt/mem_cache/radix_cache.py#L58
    """

    def __init__(self, page_size: int = 1) -> None:
        """Constructs a RadixTrie."""
        self.root = TrieNode()
        self.page_size = page_size
        self.evictable_blocks: set[BlockId] = set()
        self.all_blocks: set[BlockId] = set()

        # the lru cache contains each evictable block in the trie
        self.lru_cache = LRUCache()

    def _check_node_valid(self, node: TrieNode):
        """Rudimentary checks of data structure invariants for TrieNode."""
        if self.root == node:
            assert len(node.tokens) == 0
            assert len(node.blocks) == 0
            assert not node.parent
        else:
            assert len(node.tokens) > 0
            assert len(node.blocks) > 0
            assert node.parent
            assert len(node.tokens) % self.page_size == 0
            assert len(node.tokens) // self.page_size == len(node.blocks)

    @traced
    def insert(
        self,
        tokens: Union[np.ndarray, List[TokenId]],
        blocks: List[BlockId],
        node: Optional[TrieNode] = None,
    ) -> TrieNode:
        """Inserts `tokens` and `blocks` into the trie.

        We assume that each block contains exactly one token so the length of both
        input lists must match.

        Args:
            tokens: Tokens to insert into trie
            blocks: KV cache block for each token
            node: Node to begin insertion at. If this is not a leaf node, blocks
                  in the tree are overwritten.
        Return:
            trie_node: Node corresponding to end of the sequence where future
                       generated tokens can be inserted
        """

        if isinstance(tokens, list):
            tokens = np.array(tokens)

        def insert_helper(
            prev: TrieNode, tokens: np.ndarray, blocks: List[BlockId]
        ) -> TrieNode:
            if len(tokens) == 0:
                return prev

            key = _token_to_key(tokens, self.page_size)
            if key not in prev.children:
                # insert new node
                curr = TrieNode()
                curr.parent = prev
                curr.tokens = tokens
                curr.blocks = blocks
                prev.children[key] = curr
                prev.key_trie.insert(key)
                assert curr.is_evictable() and curr.is_leaf()
                self.evictable_blocks.update(blocks)
                self.lru_cache[curr.node_id] = curr
                self.all_blocks.update(blocks)

            curr = prev.children[key]
            prefix_len = _token_prefix_match_len(
                curr.tokens, tokens, self.page_size
            )

            if prefix_len == len(curr.tokens) and prefix_len == len(tokens):
                return curr

            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len // self.page_size :]
            if prefix_len == len(curr.tokens):
                return insert_helper(curr, unmatched_tokens, unmatched_blocks)

            # this means that we got a partial match and must split the curr node
            #   (prev) -> (curr)
            # becomes:
            #   (prev) -> (parent) -> (child)
            (parent, _) = self._split_node(curr, prefix_len)
            unmatched_tokens = tokens[prefix_len:]
            unmatched_blocks = blocks[prefix_len // self.page_size :]
            return insert_helper(parent, unmatched_tokens, unmatched_blocks)

        if len(tokens) % self.page_size != 0:
            msg = f"Insertion failed: the number of tokens is not divisible by the page size. len(tokens) == {len(tokens)} but page_size == {self.page_size}."
            raise ValueError(msg)
        if len(tokens) // self.page_size != len(blocks):
            msg = f"Insertion failed: the number of tokens and blocks do not match. len(tokens) // self.page_size == {len(tokens)} // {self.page_size} == {len(tokens) // self.page_size} but len(blocks) == {len(blocks)}."
            raise ValueError(msg)

        if node is None:
            node = self.root

        if len(tokens) == 0:
            return node

        # clone to avoid mutating the original lists
        tokens = tokens.copy()
        blocks = blocks.copy()
        return insert_helper(node, tokens, blocks)

    @traced
    def match_prefix(
        self,
        tokens: Union[np.ndarray, List[TokenId]],
        node: Optional[TrieNode] = None,
    ) -> Tuple[TrieNode, List[BlockId]]:
        """Matches the input `tokens` with the contents of the trie.

        Args:
            tokens: tokens to search the trie for
            node: Node to begin matching at.
        Return:
            Tuple containing:
                - trie_node: Node corresponding to end of matched prefix where
                             future generated tokens can be inserted.
                - block_list: KV cache blocks for matched prefix
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens)

        def match_prefix_helper(
            prev: TrieNode, tokens: np.ndarray, blocks: List[BlockId]
        ) -> TrieNode:
            if len(tokens) == 0:
                return prev

            key = _token_to_key(tokens, self.page_size)
            if key not in prev.children:
                return prev

            curr = prev.children[key]
            prefix_len = _token_prefix_match_len(
                curr.tokens, tokens, self.page_size
            )
            if prefix_len < len(curr.tokens):
                #   (prev) -> (curr)
                # becomes:
                #   (prev) -> (parent) -> (child)
                (parent, _) = self._split_node(curr, prefix_len)
                blocks.extend(parent.blocks)
                return parent
            else:
                blocks.extend(curr.blocks)
                return match_prefix_helper(curr, tokens[prefix_len:], blocks)

        if node is None:
            node = self.root
        blocks: List[BlockId] = []

        if len(tokens) == 0:
            return node, []

        tokens = tokens[: align_down(len(tokens), self.page_size)]
        curr = match_prefix_helper(node, tokens, blocks)
        return curr, blocks

    def _split_node(
        self, node: TrieNode, split_len: int
    ) -> Tuple[TrieNode, TrieNode]:
        """Splits the provided node into two.

        The resulting parent node receives exactly `split_len` tokens/blocks, and
        the child receives the remainder.

           before   │  after splitting w/ `split_len` = 2
                    │  ┌────────┐
                    │  │  ab    │ (parent)
        ┌────────┐  │  └───▲────┘
        │ abcdef │  │      │
        └────────┘  │  ┌───▼────┐
                    │  │  cdef  │ (child)
                    │  └────────┘
        """
        assert node != self.root
        assert split_len > 0
        assert split_len % self.page_size == 0

        parent = TrieNode()
        child = node
        parent.tokens, child.tokens = (
            child.tokens[:split_len],
            child.tokens[split_len:],
        )
        parent.blocks, child.blocks = (
            child.blocks[: split_len // self.page_size],
            child.blocks[split_len // self.page_size :],
        )

        parent.parent = child.parent
        assert parent.parent is not None
        assert len(parent.tokens) > 0
        parent_key = _token_to_key(parent.tokens, self.page_size)
        parent.parent.children[parent_key] = parent
        parent.parent.key_trie.insert(parent_key)
        child_key = _token_to_key(child.tokens, self.page_size)
        parent.children = {child_key: child}
        parent.key_trie.insert(child_key)
        self.lru_cache[child.node_id] = child
        child.parent = parent

        parent.active_seqs = child.active_seqs.copy()
        if parent.is_evictable():
            self.lru_cache[parent.node_id] = parent

        self._check_node_valid(parent)
        self._check_node_valid(child)
        return (parent, child)

    def mark_in_use_by(self, node: TrieNode, seq_id: SeqId):
        """Climb up the trie starting from node, marking each node as being
        in use by this seq."""

        curr = node
        while curr != self.root:
            assert curr is not None
            # optimization: if this node is already marked as using this sequence,
            # assume that it is already marked for its parents as well
            if seq_id in curr.active_seqs:
                break
            curr.active_seqs.add(seq_id)
            if not curr.is_evictable():
                self.evictable_blocks -= set(curr.blocks)
                if curr.node_id in self.lru_cache:
                    del self.lru_cache[curr.node_id]
            assert curr.parent is not None
            curr = curr.parent

    def mark_not_in_use_by(self, node: TrieNode, seq_id: SeqId):
        """Climb up the trie starting from node, marking each node as no longer
        in use by this seq. Since nodes without any users may be eligible for
        eviction, we also update its last_access_time."""

        curr = node
        while curr != self.root:
            assert curr is not None
            assert seq_id in curr.active_seqs
            curr.active_seqs.remove(seq_id)
            if curr.is_evictable():
                self.evictable_blocks.update(curr.blocks)
                self.lru_cache[curr.node_id] = curr
            assert curr.parent is not None
            curr = curr.parent

    @traced
    def evict_blocks(self, desired_num_evicted: int) -> List[BlockId]:
        """Attempt to evict at most `desired_num_evicted` blocks from trie."""
        evicted_blocks: List[BlockId] = []

        while len(evicted_blocks) < desired_num_evicted and self.lru_cache:
            leaf = self.lru_cache.pop_front()
            # we guarantee that the parent node will only be evicted after all
            # its children have been evicted. as such, this node we will evict
            # is a leaf
            assert leaf.is_evictable() and leaf.is_leaf()
            key = _token_to_key(leaf.tokens, self.page_size)
            num_blocks_to_evict = min(
                desired_num_evicted - len(evicted_blocks),
                len(leaf.blocks),
            )
            assert num_blocks_to_evict > 0
            num_tokens_to_evict = num_blocks_to_evict * self.page_size
            blocks_left, blocks_to_evict = (
                leaf.blocks[:-num_blocks_to_evict],
                leaf.blocks[-num_blocks_to_evict:],
            )
            leaf.blocks = blocks_left
            leaf.tokens = leaf.tokens[:-num_tokens_to_evict]
            evicted_blocks.extend(blocks_to_evict)
            if leaf.blocks:
                self.lru_cache[leaf.node_id] = leaf
                self.lru_cache.move_to_front(leaf.node_id)
            else:
                parent = leaf.parent
                assert parent is not None
                del parent.children[key]
                del parent.key_trie[key]

        self.evictable_blocks.difference_update(evicted_blocks)
        self.all_blocks.difference_update(evicted_blocks)
        if len(evicted_blocks) < desired_num_evicted:
            assert not self.evictable_blocks
            assert len(self.lru_cache) == 0

        return evicted_blocks

    def get_all_blocks(self) -> set[BlockId]:
        """Returns the total number of blocks in the trie."""
        return self.all_blocks

    def get_evictable_blocks(self) -> set[BlockId]:
        """Returns the number of blocks that are eligible for eviction."""
        return self.evictable_blocks

    def pretty_format(self, print_blocks: bool = False) -> List[str]:
        """Formats the contents of the trie."""

        def helper(node: TrieNode, indent: int, lines: List[str]):
            for _, child in node.children.items():
                tokens = child.tokens
                token_list = tokens.tolist()
                if print_blocks:
                    lines.append(f"{'-' * indent}{token_list} : {child.blocks}")
                else:
                    lines.append(f"{'-' * indent}{token_list}")
                helper(child, indent + 2, lines)

        lines: List[str] = []
        helper(self.root, 0, lines)
        return lines

    def pretty_print(self, print_blocks: bool = True):
        """Prints the contents of the trie."""
        for line in self.pretty_format(print_blocks):
            print(line)
