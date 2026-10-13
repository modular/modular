//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! The intrusive free list every block pool allocates from.
//!
//! A port of `_FreeKVCacheBlockQueue` in
//! `max/python/max/pipelines/kv_cache/paged_kv_cache/block_utils.py`: a doubly
//! linked list ordered least-recently-used first, with O(1) removal from the
//! middle, which is what lets a prefix hit revive one block without walking
//! the list.
//!
//! The order is the eviction order. A list starts in ascending id order;
//! a block that is allocated and then freed goes back so that:
//!
//! 1. the least recently used block is at the front, and
//! 2. of two blocks last used at the same time (freed by the same request),
//!    the one with more hashed tokens -- the tail of its chain -- is at the
//!    front, since a prefix hit needs the head of a chain before the tail.
//!    The caller keeps this by freeing a request's blocks in reverse; the
//!    list itself does not.
//!
//! A block that was never committed has nothing to lose, so pools put it at
//! the front instead, to be reused before any commit is evicted.
//!
//! The links live on the blocks, as they do in Python, but over a dense `bid`
//! rather than object references: a list is `head`/`tail` ids into the array
//! its blocks live in, and every operation takes that array.

/// A block's place in a free list -- Python's `prev_free_block` and
/// `next_free_block`, plus the membership its queue keeps in a set.
#[derive(Debug, Clone, Copy, Default)]
pub struct FreeLinks {
    pub prev: Option<usize>,
    pub next: Option<usize>,
    pub linked: bool,
}

/// A block that can sit on a [`FreeList`].
pub trait Linked {
    fn links(&self) -> &FreeLinks;
    fn links_mut(&mut self) -> &mut FreeLinks;

    /// Whether the block is on its free list, answering Python's
    /// `block in queue`. Each block belongs to at most one list -- the one
    /// over the array it lives in -- so the flag alone decides it.
    fn is_free(&self) -> bool {
        self.links().linked
    }
}

/// The ends of a free list over ids into one block array.
#[derive(Debug, Default)]
pub struct FreeList {
    pub head: Option<usize>,
    pub tail: Option<usize>,
    pub len: usize,
}

impl FreeList {
    /// A list holding `ids` in the order given.
    pub fn with_ids<T: Linked>(nodes: &mut [T], ids: impl IntoIterator<Item = usize>) -> Self {
        let mut list = FreeList::default();
        for id in ids {
            list.append_back(nodes, id);
        }
        list
    }

    pub fn contains<T: Linked>(&self, nodes: &[T], id: usize) -> bool {
        nodes[id].links().linked
    }

    /// Puts a block back at the tail, making it the most recently used.
    ///
    /// # Panics
    /// If the block is already on a list.
    pub fn append_back<T: Linked>(&mut self, nodes: &mut [T], id: usize) {
        assert!(!nodes[id].is_free(), "block {id} is already free");
        if let Some(tail) = self.tail {
            nodes[tail].links_mut().next = Some(id);
        } else {
            debug_assert!(self.head.is_none());
            self.head = Some(id);
        }
        *nodes[id].links_mut() = FreeLinks {
            prev: self.tail,
            next: None,
            linked: true,
        };
        self.tail = Some(id);
        self.len += 1;
    }

    /// Puts a block back at the head, making it the least recently used.
    ///
    /// # Panics
    /// If the block is already on a list.
    pub fn append_front<T: Linked>(&mut self, nodes: &mut [T], id: usize) {
        assert!(!nodes[id].is_free(), "block {id} is already free");
        if let Some(head) = self.head {
            nodes[head].links_mut().prev = Some(id);
        } else {
            debug_assert!(self.tail.is_none());
            self.tail = Some(id);
        }
        *nodes[id].links_mut() = FreeLinks {
            prev: None,
            next: self.head,
            linked: true,
        };
        self.head = Some(id);
        self.len += 1;
    }

    /// Unlinks a block from anywhere in the list.
    ///
    /// # Panics
    /// If the block is not on the list.
    pub fn remove<T: Linked>(&mut self, nodes: &mut [T], id: usize) {
        let FreeLinks { prev, next, linked } = *nodes[id].links();
        assert!(linked, "block {id} is not free");
        if let Some(p) = prev {
            nodes[p].links_mut().next = next;
        }
        if let Some(n) = next {
            nodes[n].links_mut().prev = prev;
        }
        if self.head == Some(id) {
            self.head = next;
        }
        if self.tail == Some(id) {
            self.tail = prev;
        }
        *nodes[id].links_mut() = FreeLinks::default();
        self.len -= 1;
    }

    /// Takes the least recently used block, or `None` if the list is empty.
    /// Python's `popleft` raises instead; a pool has to handle empty either
    /// way.
    pub fn pop_front<T: Linked>(&mut self, nodes: &mut [T]) -> Option<usize> {
        let id = self.head?;
        self.remove(nodes, id);
        Some(id)
    }

    pub fn peek_front(&self) -> Option<usize> {
        self.head
    }
}

// Ports `test_block_utils.py`, whose cases cover `test_free_block_queue.py`'s,
// from `max/tests/integration/kv_cache/jenga/` case for case. Python's
// `append` and `appendleft` are `append_back` and `append_front` here.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::block::LittleBlock;

    /// The free list never looks at a block's hash.
    type Block = LittleBlock<()>;

    /// Walks the list head to tail, checking it against the list's own state.
    ///
    /// The list is intrusive, so its bookkeeping can drift out of step with
    /// the links it maintains; every test here compares the two, which is the
    /// guarantee its callers take for granted.
    fn linked_bids(list: &FreeList, nodes: &[Block]) -> Vec<usize> {
        let mut bids = Vec::new();
        let mut cursor = list.head;
        while let Some(id) = cursor {
            assert!(
                bids.len() <= list.len,
                "list is cyclic or longer than reported"
            );
            bids.push(id);
            assert!(nodes[id].is_free(), "linked block {id} is not marked free");
            if let Some(next) = nodes[id].links().next {
                assert_eq!(nodes[next].links().prev, Some(id));
            }
            cursor = nodes[id].links().next;
        }

        assert_eq!(bids.len(), list.len);
        let mut unique = bids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), bids.len(), "block linked twice");
        if bids.is_empty() {
            assert_eq!(list.head, None);
            assert_eq!(list.tail, None);
        } else {
            assert_eq!(list.head, Some(bids[0]));
            assert_eq!(list.tail, Some(bids[bids.len() - 1]));
            assert_eq!(nodes[bids[0]].links().prev, None);
            assert_eq!(nodes[bids[bids.len() - 1]].links().next, None);
        }
        // Every block off the list must be marked so.
        for (id, node) in nodes.iter().enumerate() {
            assert_eq!(node.is_free(), bids.contains(&id), "block {id} membership");
        }
        bids
    }

    fn make_blocks(count: usize) -> Vec<Block> {
        (0..count).map(|_| Block::default()).collect()
    }

    #[test]
    fn queue_starts_in_block_id_order() {
        let mut blocks = make_blocks(4);
        let list = FreeList::with_ids(&mut blocks, 0..4);

        assert_eq!(linked_bids(&list, &blocks), [0, 1, 2, 3]);
    }

    #[test]
    fn empty_queue() {
        let mut blocks = make_blocks(1);
        let mut list = FreeList::default();

        assert_eq!(linked_bids(&list, &blocks), []);
        assert_eq!(list.pop_front(&mut blocks), None);
    }

    #[test]
    fn popleft_hands_out_the_least_recently_used_block() {
        let mut blocks = make_blocks(3);
        let mut list = FreeList::with_ids(&mut blocks, 0..3);

        let popped: Vec<usize> = (0..3)
            .map(|_| list.pop_front(&mut blocks).expect("a block"))
            .collect();

        assert_eq!(popped, [0, 1, 2]);
        assert_eq!(linked_bids(&list, &blocks), []);
    }

    #[test]
    fn append_back_puts_a_block_at_the_tail() {
        let mut blocks = make_blocks(3);
        let mut list = FreeList::with_ids(&mut blocks, 0..3);

        let bid = list.pop_front(&mut blocks).expect("a block");
        list.append_back(&mut blocks, bid);

        // Reuse makes a block the most recently used, so it is now last in line.
        assert_eq!(linked_bids(&list, &blocks), [1, 2, 0]);
    }

    #[test]
    fn append_front_puts_a_block_at_the_head() {
        let mut blocks = make_blocks(3);
        let mut list = FreeList::with_ids(&mut blocks, 0..3);

        let bid = list.pop_front(&mut blocks).expect("a block");
        list.append_front(&mut blocks, bid);

        // Unlike append_back, the block is popped again before the others.
        assert_eq!(linked_bids(&list, &blocks), [0, 1, 2]);
    }

    #[test]
    fn append_front_on_an_empty_list_becomes_both_ends() {
        let mut blocks = make_blocks(1);
        let mut list = FreeList::default();

        list.append_front(&mut blocks, 0);

        assert_eq!(list.head, Some(0));
        assert_eq!(list.tail, Some(0));
        assert_eq!(linked_bids(&list, &blocks), [0]);
    }

    #[test]
    fn append_front_and_append_back_interleave_correctly() {
        let mut blocks = make_blocks(4);
        let mut list = FreeList::default();

        list.append_back(&mut blocks, 0);
        list.append_front(&mut blocks, 1);
        list.append_back(&mut blocks, 2);
        list.append_front(&mut blocks, 3);

        assert_eq!(linked_bids(&list, &blocks), [3, 1, 0, 2]);
    }

    #[test]
    fn remove_from_the_middle() {
        let mut blocks = make_blocks(4);
        let mut list = FreeList::with_ids(&mut blocks, 0..4);

        list.remove(&mut blocks, 1);
        list.remove(&mut blocks, 2);

        assert_eq!(linked_bids(&list, &blocks), [0, 3]);
        assert_eq!(blocks[1].links().prev, None);
        assert_eq!(blocks[1].links().next, None);
    }

    #[test]
    fn remove_from_an_end() {
        for id in [0, 3] {
            let mut blocks = make_blocks(4);
            let mut list = FreeList::with_ids(&mut blocks, 0..4);

            list.remove(&mut blocks, id);

            assert!(!linked_bids(&list, &blocks).contains(&id));
            assert_eq!(list.len, 3);
        }
    }

    #[test]
    fn remove_the_last_block_empties_the_queue() {
        let mut blocks = make_blocks(1);
        let mut list = FreeList::with_ids(&mut blocks, 0..1);

        list.remove(&mut blocks, 0);

        assert_eq!(linked_bids(&list, &blocks), []);
        list.append_back(&mut blocks, 0);
        assert_eq!(linked_bids(&list, &blocks), [0]);
    }

    #[test]
    fn membership_tracks_the_list() {
        let mut blocks = make_blocks(2);
        let mut list = FreeList::with_ids(&mut blocks, 0..2);

        assert!(blocks[0].is_free());
        list.remove(&mut blocks, 0);
        assert!(!blocks[0].is_free());
        list.append_back(&mut blocks, 0);
        assert!(blocks[0].is_free());
    }

    #[test]
    fn queues_over_the_same_block_ids_are_independent() {
        // Two pools of the same size hold different blocks under the same ids,
        // so membership follows the array a list is over, not the id alone.
        let mut first_blocks = make_blocks(2);
        let mut first = FreeList::with_ids(&mut first_blocks, 0..2);
        let mut second_blocks = make_blocks(2);
        let second = FreeList::with_ids(&mut second_blocks, 0..2);

        first.pop_front(&mut first_blocks);

        assert_eq!(linked_bids(&first, &first_blocks), [1]);
        assert_eq!(linked_bids(&second, &second_blocks), [0, 1]);
    }

    #[test]
    fn churn_preserves_the_list() {
        // Interleaved removes and appends, the pattern a pool actually produces.
        let mut blocks = make_blocks(6);
        let mut list = FreeList::with_ids(&mut blocks, 0..6);
        let mut live: Vec<usize> = Vec::new();

        for _ in 0..3 {
            for _ in 0..4 {
                live.push(list.pop_front(&mut blocks).expect("a block"));
                linked_bids(&list, &blocks);
            }
            // Requests free their blocks tail first, which is why a pool
            // reverses them before handing them back.
            while let Some(bid) = live.pop() {
                list.append_back(&mut blocks, bid);
                linked_bids(&list, &blocks);
            }
        }

        assert_eq!(list.len, 6);
    }
}
