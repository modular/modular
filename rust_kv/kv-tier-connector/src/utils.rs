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
use crate::error::ConnectorError;

/// Canonical block hash bytes (8 for ahash64-family, 32 for full SHA-256).
/// Either fits in a pair of u128s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u128, pub u128);

impl TryFrom<&[u8]> for BlockHash {
    type Error = ConnectorError;

    /// Packs `kv_hash_algo`'s raw hash bytes losslessly into the two halves --
    /// an 8-byte ahash64-family hash zero-extends into the first half, and a
    /// 32-byte SHA-256 hash splits evenly across both. Never persisted or
    /// compared across processes (the disk tier is wiped on `shutdown`), so
    /// the packing only needs to be lossless and stable within one process.
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        match bytes.len() {
            8 => {
                let mut lo = [0u8; 16];
                lo[..8].copy_from_slice(bytes);
                Ok(BlockHash(u128::from_le_bytes(lo), 0))
            }
            32 => {
                let mut hi = [0u8; 16];
                let mut lo = [0u8; 16];
                hi.copy_from_slice(&bytes[..16]);
                lo.copy_from_slice(&bytes[16..]);
                Ok(BlockHash(u128::from_le_bytes(hi), u128::from_le_bytes(lo)))
            }
            n => Err(ConnectorError::Config(format!(
                "block hash must be 8 or 32 bytes, found {n}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeafHash {
    pub leaf_idx: usize,
    pub block_hash: BlockHash,
}

impl LeafHash {
    pub fn to_hex(&self) -> String {
        format!(
            "{:x}-{:x}{:x}",
            self.leaf_idx, self.block_hash.0, self.block_hash.1,
        )
    }
}
