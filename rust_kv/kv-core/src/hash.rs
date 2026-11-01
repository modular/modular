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
//! Chained block hashing for the prefix cache.
//!
//! A request's tokens are cut into whole blocks of `block_size` tokens (a
//! trailing partial block is dropped) and each block is hashed together with
//! the hash of the block before it, so a block hash identifies the whole
//! prefix up to and including that block. The hashes key the device prefix
//! cache and are persisted by the dKV and disk tiers, so their bytes are an
//! external contract: this module reproduces, bit for bit, what MAX's former
//! Mojo hasher produced. It was checked against that hasher over a grid of
//! seeds, salts, block sizes, parents, dtypes and overrides before it was
//! deleted.
//!
//! Two families exist, both over tokens serialized as little-endian `int32`:
//!
//! - [`HashAlgo::Ahash64`]: a 64-bit, non-cryptographic hash. Each block is
//!   hashed with aHash (an in-crate port of Mojo's `_ahash.mojo`) keyed by a
//!   32-byte seed, then chained as `ahash(prev, block_hash)` under the same
//!   seed. Hashes are 8 bytes, the big-endian two's-complement encoding of the
//!   `u64`. The first `prev` is the 8-byte parent hash, zero by default.
//! - [`HashAlgo::Sha256`]: each block hashes to `sha256(sha256(tokens) ||
//!   prev)`. Hashes are 32 bytes. The first `prev` is the 32-byte parent hash,
//!   which defaults to the root derived from the seed and salt.
//!
//! The root derivation is `(seed or zeros) XOR sha256(salt)`, with an empty
//! salt meaning no salt. For SHA-256 the 32 bytes are the chain root. For
//! aHash they are the key material XORed into the hasher's initial state.
//! This is the mach `block-hash` scheme minus its Go loop bound: mach's copy
//! only agrees with MAX at block sizes of at most four tokens or multiples of
//! four, because it ports a Go bug. This module follows Mojo at every block
//! size.
//!
//! The module is pure: no I/O, no allocation beyond the output.

use std::{borrow::Cow, fmt};

use sha2::{Digest, Sha256};

/// Bytes of a seed or of a SHA-256 hash.
pub const SEED_LEN: usize = 32;
/// Bytes of an [`HashAlgo::Ahash64`] block hash.
pub const AHASH64_LEN: usize = 8;
/// Bytes of a [`HashAlgo::Sha256`] block hash.
pub const SHA256_LEN: usize = 32;

/// The hash family used to chain block hashes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashAlgo {
    /// Seeded, chained aHash. 8-byte hashes.
    Ahash64,
    /// Chained SHA-256. 32-byte hashes.
    Sha256,
}

impl HashAlgo {
    /// Bytes in one block hash of this family.
    #[must_use]
    pub const fn hash_len(self) -> usize {
        match self {
            Self::Ahash64 => AHASH64_LEN,
            Self::Sha256 => SHA256_LEN,
        }
    }
}

/// Why a hashing request was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashError {
    /// `block_size` was zero.
    ZeroBlockSize,
    /// The token width was not 4 or 8 bytes.
    UnsupportedTokenWidth(usize),
    /// The token buffer is not a whole number of tokens.
    RaggedTokens {
        /// Bytes supplied.
        len: usize,
        /// Bytes per token.
        width: usize,
    },
    /// A seed was not [`SEED_LEN`] bytes.
    BadSeedLen(usize),
    /// A parent hash had the wrong length for the algorithm.
    BadParentLen {
        /// The algorithm the parent was given for.
        algo: HashAlgo,
        /// Bytes supplied.
        len: usize,
    },
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockSize => write!(f, "block_size must be greater than 0"),
            Self::UnsupportedTokenWidth(w) => {
                write!(f, "tokens must be 4 or 8 bytes wide, got {w}")
            }
            Self::RaggedTokens { len, width } => {
                write!(
                    f,
                    "{len} token bytes is not a multiple of the token width {width}"
                )
            }
            Self::BadSeedLen(len) => {
                write!(f, "seed must be exactly {SEED_LEN} bytes, got {len}")
            }
            Self::BadParentLen { algo, len } => write!(
                f,
                "algo={} requires {}-byte parent_hash, got {len}",
                match algo {
                    HashAlgo::Ahash64 => "ahash64",
                    HashAlgo::Sha256 => "sha256",
                },
                algo.hash_len(),
            ),
        }
    }
}

impl std::error::Error for HashError {}

/// One media placeholder: while hashing, the token at `index` is read as
/// `value` (its low 32 bits) instead of its stored value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenHashOverride {
    /// Index into the token slice being hashed. Out of range is ignored.
    pub index: usize,
    /// The content hash standing in for the token, as raw `int32` bits.
    pub value: u32,
}

/// The block hashes of one request: `len()` hashes of `algo.hash_len()` bytes,
/// stored back to back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockHashes {
    stride: usize,
    bytes: Vec<u8>,
}

impl BlockHashes {
    /// Number of block hashes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len().checked_div(self.stride).unwrap_or(0)
    }

    /// Whether there are no block hashes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Bytes per hash.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The hashes, oldest block first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.bytes.chunks_exact(self.stride.max(1))
    }

    /// All hashes concatenated.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Combines a cluster-level `seed` and a per-request `salt` into 32 bytes:
/// `(seed or zeros) XOR sha256(salt)`. An absent or empty salt leaves the
/// seed unchanged.
#[must_use]
pub fn root_parent_hash(seed: Option<&[u8; SEED_LEN]>, salt: Option<&str>) -> [u8; SEED_LEN] {
    let mut root = seed.copied().unwrap_or([0; SEED_LEN]);
    if let Some(salt) = salt.filter(|s| !s.is_empty()) {
        let digest = Sha256::digest(salt.as_bytes());
        for (r, d) in root.iter_mut().zip(digest) {
            *r ^= d;
        }
    }
    root
}

// --- aHash ------------------------------------------------------------------

/// The PCG multiplier from `_ahash.mojo`.
const MULTIPLE: u64 = 6_364_136_223_846_793_005;
/// The rotation from `_ahash.mojo`.
const ROT: u32 = 23;
/// The pi constants folded into the key.
const PI_KEY: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

#[inline(always)]
fn folded_multiply(lhs: u64, rhs: u64) -> u64 {
    let m = u128::from(lhs) * u128::from(rhs);
    #[expect(clippy::cast_possible_truncation, reason = "taking both halves")]
    let (lo, hi) = (m as u64, (m >> 64) as u64);
    lo ^ hi
}

#[inline(always)]
fn read_u64(data: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&data[at..at + 8]);
    u64::from_le_bytes(b)
}

#[inline(always)]
fn read_u32(data: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[at..at + 4]);
    u64::from(u32::from_le_bytes(b))
}

/// aHash with a 256-bit key, mirroring Mojo's `AHasher[U256(0)](seed)`.
#[derive(Clone, Copy)]
struct AHasher {
    buffer: u64,
    pad: u64,
    extra: [u64; 2],
}

impl AHasher {
    fn new(seed: &[u64; 4]) -> Self {
        let k = [
            PI_KEY[0] ^ seed[0],
            PI_KEY[1] ^ seed[1],
            PI_KEY[2] ^ seed[2],
            PI_KEY[3] ^ seed[3],
        ];
        Self {
            buffer: k[0],
            pad: k[1],
            extra: [k[2], k[3]],
        }
    }

    #[inline(always)]
    fn large_update(&mut self, a: u64, b: u64) {
        let combined = folded_multiply(a ^ self.extra[0], b ^ self.extra[1]);
        self.buffer = (self.buffer.wrapping_add(self.pad) ^ combined).rotate_left(ROT);
    }

    /// Hashes a byte string, as Mojo's `hash_seeded_bytes` does.
    #[inline(always)]
    fn update(&mut self, data: &[u8]) {
        let len = data.len();
        self.buffer = self.buffer.wrapping_add(len as u64).wrapping_mul(MULTIPLE);
        if len > 8 {
            if len > 16 {
                // The last 16 bytes first, then every full 16-byte group from
                // the front; for lengths that are not a multiple of 16 the
                // tail overlaps the last group.
                self.large_update(read_u64(data, len - 16), read_u64(data, len - 8));
                let mut offset = 0;
                while len - offset > 16 {
                    self.large_update(read_u64(data, offset), read_u64(data, offset + 8));
                    offset += 16;
                }
            } else {
                self.large_update(read_u64(data, 0), read_u64(data, len - 8));
            }
        } else if len >= 4 {
            self.large_update(read_u32(data, 0), read_u32(data, len - 4));
        } else if len >= 2 {
            let a = u64::from(u16::from_le_bytes([data[0], data[1]]));
            self.large_update(a, u64::from(data[len - 1]));
        } else if len == 1 {
            self.large_update(u64::from(data[0]), u64::from(data[0]));
        } else {
            self.large_update(0, 0);
        }
    }

    #[inline(always)]
    fn finish(self) -> u64 {
        let rot = (self.buffer & 63) as u32;
        folded_multiply(self.buffer, self.pad).rotate_left(rot)
    }
}

/// aHash of `data` under `seed`.
fn ahash_bytes(seed: &[u64; 4], data: &[u8]) -> u64 {
    let mut h = AHasher::new(seed);
    h.update(data);
    h.finish()
}

/// aHash of the pair `(a, b)` under `seed`, as Mojo hashes a `SIMD[uint64, 2]`:
/// one wide round and no length mix-in.
fn ahash_pair(seed: &[u64; 4], a: u64, b: u64) -> u64 {
    let mut h = AHasher::new(seed);
    h.large_update(a, b);
    h.finish()
}

fn seed_lanes(seed: &[u8; SEED_LEN]) -> [u64; 4] {
    let mut lanes = [0u64; 4];
    for (lane, chunk) in lanes.iter_mut().zip(seed.chunks_exact(8)) {
        let mut b = [0u8; 8];
        b.copy_from_slice(chunk);
        *lane = u64::from_le_bytes(b);
    }
    lanes
}

/// Chained aHash over `tokens` (little-endian `int32` bytes), appending one
/// 8-byte big-endian hash per whole block to `out`.
fn chain_ahash64(
    tokens: &[u8],
    block_bytes: usize,
    parent: u64,
    seed: &[u8; SEED_LEN],
    out: &mut Vec<u8>,
) {
    let lanes = seed_lanes(seed);
    let mut prev = parent;
    for block in tokens.chunks_exact(block_bytes) {
        let token_hash = ahash_bytes(&lanes, block);
        prev = ahash_pair(&lanes, prev, token_hash);
        out.extend_from_slice(&prev.to_be_bytes());
    }
}

/// Chained SHA-256 over `tokens`, appending one 32-byte hash per whole block.
fn chain_sha256(tokens: &[u8], block_bytes: usize, parent: &[u8; SHA256_LEN], out: &mut Vec<u8>) {
    let mut prev = *parent;
    for block in tokens.chunks_exact(block_bytes) {
        let local = Sha256::digest(block);
        let mut pair = Sha256::new();
        pair.update(local);
        pair.update(prev);
        prev = pair.finalize().into();
        out.extend_from_slice(&prev);
    }
}

/// A request to hash the whole blocks of a token buffer, chained from a parent.
///
/// - `tokens` holds the tokens back to back, little-endian, `token_width`
///   bytes each. Only 4 (`int32`/`uint32`) and 8 (`int64`/`uint64`) are
///   accepted, and 8-byte tokens are narrowed to their low 32 bits, as numpy's
///   `astype(int32)` wraps. Callers with narrower dtypes widen them first.
/// - `parent` is the previous block's hash, or `None` for the first block:
///   eight zero bytes for [`HashAlgo::Ahash64`], and the seed/salt root for
///   [`HashAlgo::Sha256`].
/// - `seed` is the 32-byte cluster seed and `salt` the per-request salt. See
///   [`root_parent_hash`]. For aHash the root is the hasher key; for SHA-256 it
///   is only the default parent.
/// - `overrides` replace tokens by index while hashing. If several name the
///   same index, the last wins; callers that consider that an error check for
///   it first.
#[derive(Clone, Copy, Debug)]
pub struct BlockHashRequest<'a> {
    /// The hash family.
    pub algo: HashAlgo,
    /// The token bytes.
    pub tokens: &'a [u8],
    /// Bytes per token, 4 or 8.
    pub token_width: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// The previous block's hash.
    pub parent: Option<&'a [u8]>,
    /// The 32-byte cluster seed.
    pub seed: Option<&'a [u8]>,
    /// The per-request salt.
    pub salt: Option<&'a str>,
    /// Placeholder tokens to replace while hashing.
    pub overrides: &'a [TokenHashOverride],
}

/// Hashes the whole blocks of `req.tokens`, chained from `req.parent`.
///
/// # Errors
///
/// Rejects a zero block size, an unsupported or ragged token buffer, a seed
/// that is not 32 bytes and a parent of the wrong length.
pub fn hash_blocks(req: &BlockHashRequest<'_>) -> Result<BlockHashes, HashError> {
    let BlockHashRequest {
        algo,
        tokens,
        token_width,
        block_size,
        parent,
        seed,
        salt,
        overrides,
    } = *req;
    if block_size == 0 {
        return Err(HashError::ZeroBlockSize);
    }
    if token_width != 4 && token_width != 8 {
        return Err(HashError::UnsupportedTokenWidth(token_width));
    }
    if !tokens.len().is_multiple_of(token_width) {
        return Err(HashError::RaggedTokens {
            len: tokens.len(),
            width: token_width,
        });
    }
    let seed: Option<&[u8; SEED_LEN]> = match seed {
        None => None,
        Some(s) => Some(s.try_into().map_err(|_| HashError::BadSeedLen(s.len()))?),
    };
    let root = root_parent_hash(seed, salt);

    // Canonical image: little-endian int32 per token. Borrowed when the input
    // already is one and nothing is overridden.
    let image: Cow<'_, [u8]> = if token_width == 4 && overrides.is_empty() {
        Cow::Borrowed(tokens)
    } else {
        let mut owned = Vec::with_capacity(tokens.len() / token_width * 4);
        for token in tokens.chunks_exact(token_width) {
            owned.extend_from_slice(&token[..4]);
        }
        for o in overrides {
            let start = o.index.saturating_mul(4);
            if let Some(slot) = owned.get_mut(start..start.saturating_add(4)) {
                slot.copy_from_slice(&o.value.to_le_bytes());
            }
        }
        Cow::Owned(owned)
    };

    let block_bytes = block_size.saturating_mul(4);
    let stride = algo.hash_len();
    let mut out = Vec::with_capacity(image.len() / block_bytes * stride);
    match algo {
        HashAlgo::Ahash64 => {
            let parent = match parent {
                None => 0,
                Some(p) => {
                    let p: [u8; AHASH64_LEN] = p
                        .try_into()
                        .map_err(|_| HashError::BadParentLen { algo, len: p.len() })?;
                    u64::from_be_bytes(p)
                }
            };
            chain_ahash64(&image, block_bytes, parent, &root, &mut out);
        }
        HashAlgo::Sha256 => {
            let parent: [u8; SHA256_LEN] = match parent {
                None => root,
                Some(p) => p
                    .try_into()
                    .map_err(|_| HashError::BadParentLen { algo, len: p.len() })?,
            };
            chain_sha256(&image, block_bytes, &parent, &mut out);
        }
    }
    Ok(BlockHashes { stride, bytes: out })
}

#[cfg(test)]
mod tests;
