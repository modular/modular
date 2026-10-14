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
//! Byte offsets into the pinned host staging region, resolved to addresses.
//!
//! MAX hands the region over as one or more chunks, each a whole number of
//! huge blocks. A single registered mapping is the one-chunk case; pinned
//! memory allocated by the GPU driver arrives as many separately allocated
//! chunks. Every leaf's row divides a huge block, so no row straddles two
//! chunks.

use crate::{
    disk_tier::DIRECT_IO_ALIGN,
    error::{ConnectorError, Result},
};

/// The host staging region as `chunk_bases.len()` chunks of `chunk_bytes`.
#[derive(Clone, Debug)]
pub struct HostRegion {
    chunk_bases: Vec<usize>,
    chunk_bytes: usize,
}

impl HostRegion {
    /// Describes `num_huge_blocks` huge blocks laid out `rows_per_chunk` to a
    /// chunk, chunk `i` starting at `chunk_bases[i]`.
    ///
    /// A huge block is `cache_ratios[l] * bytes_per_leaf[l]` bytes, which has
    /// to agree across leaves.
    pub fn new(
        chunk_bases: Vec<usize>,
        rows_per_chunk: usize,
        num_huge_blocks: usize,
        bytes_per_leaf: &[usize],
        cache_ratios: &[usize],
    ) -> Result<Self> {
        let config = |msg: String| Err(ConnectorError::Config(msg));
        if rows_per_chunk == 0 {
            return config("host_rows_per_chunk must be positive".to_owned());
        }
        if bytes_per_leaf.len() != cache_ratios.len() || bytes_per_leaf.is_empty() {
            return config(format!(
                "bytes_per_leaf ({}) and cache_ratios ({}) must name the same, non-zero number of leaves",
                bytes_per_leaf.len(),
                cache_ratios.len()
            ));
        }
        let huge_block_bytes = bytes_per_leaf[0] * cache_ratios[0];
        if huge_block_bytes == 0
            || bytes_per_leaf
                .iter()
                .zip(cache_ratios)
                .any(|(bytes, ratio)| bytes * ratio != huge_block_bytes)
        {
            return config(format!(
                "leaves disagree on the huge block size: bytes_per_leaf {bytes_per_leaf:?} x cache_ratios {cache_ratios:?}"
            ));
        }
        let num_chunks = num_huge_blocks.div_ceil(rows_per_chunk);
        if chunk_bases.len() != num_chunks {
            return config(format!(
                "{num_huge_blocks} huge blocks at {rows_per_chunk} per chunk need {num_chunks} host chunk addresses, got {}",
                chunk_bases.len()
            ));
        }
        Ok(HostRegion {
            chunk_bases,
            chunk_bytes: rows_per_chunk * huge_block_bytes,
        })
    }

    /// The host address of region byte `offset`, the start of a `len`-byte
    /// span that must not cross a chunk boundary.
    ///
    /// Checked in every build, not just debug: a span that crosses a chunk
    /// boundary would otherwise resolve to a silently wrong address inside
    /// someone else's chunk, corrupting the KV cache rather than failing
    /// loudly. The branch sits on a path that is already setting up a DMA, so
    /// its cost disappears next to the copy it guards.
    pub fn addr(&self, offset: usize, len: usize) -> usize {
        let within = offset % self.chunk_bytes;
        assert!(
            within + len <= self.chunk_bytes,
            "host span [{offset}, {}) crosses a {}-byte chunk boundary",
            offset + len,
            self.chunk_bytes
        );
        self.chunk_bases[offset / self.chunk_bytes] + within
    }

    /// Checks that every chunk can take direct I/O: each chunk's base address
    /// is `DIRECT_IO_ALIGN`-aligned, and a chunk spans a whole, non-zero
    /// number of `DIRECT_IO_ALIGN` units.
    ///
    /// # Errors
    ///
    /// Returns a config error naming the first chunk that fails.
    pub fn check_direct_io(&self) -> Result<()> {
        if self.chunk_bytes == 0 || !self.chunk_bytes.is_multiple_of(DIRECT_IO_ALIGN) {
            return Err(ConnectorError::Config(format!(
                "use_disk_direct_io needs host chunks that are a positive multiple of \
                 {DIRECT_IO_ALIGN} bytes, got {} bytes",
                self.chunk_bytes
            )));
        }
        if let Some((idx, base)) = self
            .chunk_bases
            .iter()
            .enumerate()
            .find(|(_, base)| !base.is_multiple_of(DIRECT_IO_ALIGN))
        {
            return Err(ConnectorError::Config(format!(
                "use_disk_direct_io needs {DIRECT_IO_ALIGN}-byte aligned host staging, \
                 but chunk {idx} starts at {base:#x}"
            )));
        }
        Ok(())
    }

    /// The host address of every chunk, in order: each chunk's first byte. The
    /// disk tier's direct-I/O probe runs against the last chunk's base.
    #[must_use]
    pub fn chunk_bases(&self) -> &[usize] {
        &self.chunk_bases
    }
}

#[cfg(test)]
mod tests {
    use super::HostRegion;

    const HUGE: usize = 6 * 1024;

    /// Two leaves of 2 KiB and 3 KiB rows: both tile a 6 KiB huge block.
    fn two_leaves() -> (Vec<usize>, Vec<usize>) {
        (vec![2 * 1024, 3 * 1024], vec![3, 2])
    }

    #[test]
    fn one_chunk_is_the_contiguous_region() {
        let (bytes, ratios) = two_leaves();
        let base = 0x7f00_0000_0000;
        let region = HostRegion::new(vec![base], 10, 10, &bytes, &ratios).expect("a region");
        for offset in [0, 1, HUGE - 1, HUGE, 7 * HUGE + 123, 10 * HUGE - 1] {
            assert_eq!(region.addr(offset, 1), base + offset);
        }
    }

    #[test]
    fn chunks_resolve_to_their_own_base() {
        let (bytes, ratios) = two_leaves();
        let bases = vec![0x1000_0000, 0x9000_0000, 0x5000_0000];
        // 7 huge blocks at 3 per chunk: chunks of 3, 3 and 1.
        let region = HostRegion::new(bases.clone(), 3, 7, &bytes, &ratios).expect("a region");
        let chunk = 3 * HUGE;
        assert_eq!(region.addr(0, HUGE), bases[0]);
        assert_eq!(region.addr(chunk - HUGE, HUGE), bases[0] + chunk - HUGE);
        assert_eq!(region.addr(chunk, HUGE), bases[1]);
        assert_eq!(region.addr(2 * chunk + 5, 1), bases[2] + 5);
        assert_eq!(region.addr(7 * HUGE - 1, 1), bases[2] + HUGE - 1);
    }

    #[test]
    fn every_leaf_row_lands_inside_one_chunk() {
        let (bytes, ratios) = two_leaves();
        let bases = vec![0x1000_0000, 0x9000_0000, 0x5000_0000, 0x3000_0000];
        let num_huge_blocks = 7;
        let region =
            HostRegion::new(bases.clone(), 2, num_huge_blocks, &bytes, &ratios).expect("a region");
        let chunk = 2 * HUGE;
        for (leaf_bytes, ratio) in bytes.iter().zip(&ratios) {
            for host_bid in 0..num_huge_blocks * ratio {
                let offset = host_bid * leaf_bytes;
                let addr = region.addr(offset, *leaf_bytes);
                let base = bases[offset / chunk];
                assert!(addr >= base && addr + leaf_bytes <= base + chunk);
            }
        }
    }

    #[test]
    fn rejects_a_chunk_count_that_does_not_cover_the_blocks() {
        let (bytes, ratios) = two_leaves();
        assert!(HostRegion::new(vec![1, 2], 3, 7, &bytes, &ratios).is_err());
        assert!(HostRegion::new(vec![1, 2, 3, 4], 3, 7, &bytes, &ratios).is_err());
    }

    #[test]
    fn rejects_zero_rows_per_chunk() {
        let (bytes, ratios) = two_leaves();
        assert!(HostRegion::new(vec![], 0, 0, &bytes, &ratios).is_err());
    }

    #[test]
    fn rejects_leaves_that_disagree_on_the_huge_block() {
        assert!(HostRegion::new(vec![1], 4, 4, &[2048, 3072], &[3, 3]).is_err());
        assert!(HostRegion::new(vec![1], 4, 4, &[2048], &[3, 2]).is_err());
    }

    #[test]
    #[should_panic(expected = "crosses a")]
    fn addr_panics_on_a_span_crossing_a_chunk_boundary_in_release_builds_too() {
        let (bytes, ratios) = two_leaves();
        let bases = vec![0x1000_0000, 0x9000_0000];
        let region = HostRegion::new(bases, 3, 6, &bytes, &ratios).expect("a region");
        let chunk = 3 * HUGE;
        // One byte short of the chunk boundary, spanning a whole huge block:
        // this must panic rather than silently resolve into the next chunk.
        let _ = region.addr(chunk - 1, HUGE);
    }

    #[test]
    fn direct_io_check_rejects_short_chunks_and_unaligned_bases() {
        // Two chunks of two 4 KiB rows each.
        let ok = HostRegion::new(vec![0x10000, 0x12000], 2, 4, &[4096], &[1]).unwrap();
        ok.check_direct_io().expect("aligned chunks pass");

        let unaligned = HostRegion::new(vec![0x10000, 0x12010], 2, 4, &[4096], &[1]).unwrap();
        let err = unaligned
            .check_direct_io()
            .expect_err("chunk 1 is unaligned");
        assert!(err.to_string().contains("chunk 1"), "{err}");

        let short = HostRegion::new(vec![0x10000], 1, 1, &[16], &[1]).unwrap();
        short
            .check_direct_io()
            .expect_err("a 16-byte chunk is too short");
    }
}
