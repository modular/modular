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
//! The two kinds of block a [`crate::pool::block_pool::JengaBlockPool`] holds.
//!
//! Ported from MAX's Python `HugeKVCacheBlock` and `LittleKVCacheBlock`, now
//! deleted. The Python blocks carried their own `bid`, `cache_id` and `huge_block`; here a
//! block is found at `little_blocks[cache][bid]`, so all three are known from
//! where it sits and the block holds only what it cannot derive. MAX's block
//! objects are now handles on a `(cache, bid)` pair
//! (`max._kv_core.LittleKVCacheBlock`).

use crate::pool::free_list::{FreeLinks, Linked};

/// One cache's page. Its `bid` is the page index into that cache's view of the
/// pool buffer -- the number the kernel's lookup table holds -- so the huge
/// block backing it is `bid / ratio`.
#[derive(Debug)]
pub struct LittleBlock<H> {
    /// How many requests hold this block.
    pub ref_cnt: i64,
    /// The hash this block is committed under, once it is full.
    pub block_hash: Option<H>,
    /// Its place in a [`crate::pool::free_list::FreeQueue`], while on one.
    pub free: Option<u64>,
}

// Derived, this would demand `H: Default`, which a hash need not be.
impl<H> Default for LittleBlock<H> {
    fn default() -> Self {
        LittleBlock {
            ref_cnt: 0,
            block_hash: None,
            free: None,
        }
    }
}

impl<H> LittleBlock<H> {
    /// Whether the block is on a free queue.
    pub fn is_free(&self) -> bool {
        self.free.is_some()
    }
}

/// A run of bytes every cache can tile, serving one of them at a time.
#[derive(Debug, Default)]
pub struct HugeBlock {
    /// The cache currently allowed to allocate from these bytes, or `None`
    /// while they have never been claimed (or hold the null block). It
    /// outlives parking, so whoever claims the block next knows whose commits
    /// its bytes still back.
    pub little_block_type: Option<usize>,
    /// How many of `little_block_type`'s little blocks in here a request holds.
    /// Zero is what parks the block, so the bytes are up for grabs.
    pub num_referenced: usize,
    pub free: FreeLinks,
}

impl Linked for HugeBlock {
    fn links(&self) -> &FreeLinks {
        &self.free
    }
    fn links_mut(&mut self) -> &mut FreeLinks {
        &mut self.free
    }
}
