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
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code"
)]

use super::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A small deterministic generator, so the property tests need no crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for b in &mut out {
            *b = self.next().to_le_bytes()[3];
        }
        out
    }

    fn tokens(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (self.next() as u32).to_le_bytes())
            .collect()
    }
}

fn run(
    algo: HashAlgo,
    tokens: &[u8],
    block_size: usize,
    parent: Option<&[u8]>,
    seed: Option<&[u8]>,
    salt: Option<&str>,
) -> BlockHashes {
    hash_blocks(&BlockHashRequest {
        algo,
        tokens,
        token_width: 4,
        block_size,
        parent,
        seed,
        salt,
        overrides: &[],
    })
    .unwrap()
}

#[test]
fn split_sequence_with_chained_parent_equals_whole() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..300 {
        let algo = if rng.below(2) == 0 {
            HashAlgo::Ahash64
        } else {
            HashAlgo::Sha256
        };
        let block_size = 1 + rng.below(40);
        let blocks = rng.below(12);
        let extra = rng.below(block_size);
        let tokens = rng.tokens(blocks * block_size + extra);
        let seed: Option<[u8; 32]> = (rng.below(2) == 0).then(|| rng.bytes());
        let salt = ["", "a", "tenant-7"][rng.below(3)];
        let (seed, salt) = (seed.as_ref().map(<[u8; 32]>::as_slice), Some(salt));

        let whole = run(algo, &tokens, block_size, None, seed, salt);
        let split = rng.below(blocks + 1);
        let head = run(
            algo,
            &tokens[..split * block_size * 4],
            block_size,
            None,
            seed,
            salt,
        );
        let parent = head.iter().last();
        let tail = run(
            algo,
            &tokens[split * block_size * 4..],
            block_size,
            parent,
            seed,
            salt,
        );

        let mut joined = head.as_bytes().to_vec();
        joined.extend_from_slice(tail.as_bytes());
        assert_eq!(
            joined,
            whole.as_bytes(),
            "{algo:?} bs={block_size} split={split}"
        );
        assert_eq!(whole.len(), tokens.len() / 4 / block_size);
    }
}

#[test]
fn every_token_and_position_matters() {
    // A block hash must depend on every token in the block, including at
    // block sizes that are not multiples of four, where mach's port drops
    // some.
    let mut rng = Rng(7);
    for block_size in 1..=70 {
        let tokens = rng.tokens(block_size * 2);
        for algo in [HashAlgo::Ahash64, HashAlgo::Sha256] {
            let base = run(algo, &tokens, block_size, None, None, None);
            for t in 0..block_size * 2 {
                let mut flipped = tokens.clone();
                flipped[t * 4] ^= 1;
                let other = run(algo, &flipped, block_size, None, None, None);
                let block = t / block_size;
                assert_ne!(
                    base.iter().nth(block),
                    other.iter().nth(block),
                    "{algo:?} bs={block_size} token {t} is ignored"
                );
            }
        }
    }
}

#[test]
fn overrides_equal_patched_tokens() {
    let mut rng = Rng(11);
    for _ in 0..100 {
        let block_size = 1 + rng.below(20);
        let n = rng.below(10 * block_size) + 1;
        let tokens = rng.tokens(n);
        let overrides: Vec<TokenHashOverride> = (0..rng.below(4))
            .map(|i| TokenHashOverride {
                index: i * 3 + rng.below(3),
                value: rng.next() as u32,
            })
            .collect();
        let mut patched = tokens.clone();
        for o in &overrides {
            if o.index < n {
                patched[o.index * 4..o.index * 4 + 4].copy_from_slice(&o.value.to_le_bytes());
            }
        }
        for algo in [HashAlgo::Ahash64, HashAlgo::Sha256] {
            let got = hash_blocks(&BlockHashRequest {
                algo,
                tokens: &tokens,
                token_width: 4,
                block_size,
                parent: None,
                seed: None,
                salt: None,
                overrides: &overrides,
            })
            .unwrap();
            assert_eq!(got, run(algo, &patched, block_size, None, None, None));
        }
    }
}

#[test]
fn wide_tokens_are_narrowed_like_astype() {
    let wide: Vec<u8> = [1u64 << 32 | 5, 1 << 31, u64::MAX, 7, 9, 11, 13, 15]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let narrow: Vec<u8> = [5u32, 1 << 31, u32::MAX, 7, 9, 11, 13, 15]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let got = hash_blocks(&BlockHashRequest {
        algo: HashAlgo::Ahash64,
        tokens: &wide,
        token_width: 8,
        block_size: 3,
        parent: None,
        seed: None,
        salt: None,
        overrides: &[],
    })
    .unwrap();
    assert_eq!(got, run(HashAlgo::Ahash64, &narrow, 3, None, None, None));
}

#[test]
fn root_parent_hash_derivation() {
    assert_eq!(root_parent_hash(None, None), [0; 32]);
    assert_eq!(root_parent_hash(None, Some("")), [0; 32]);
    let seed = [0xA5u8; 32];
    assert_eq!(root_parent_hash(Some(&seed), Some("")), seed);
    // sha256("abc") from FIPS 180-4.
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(hex(&root_parent_hash(None, Some("abc"))), abc);
    let mixed = root_parent_hash(Some(&seed), Some("abc"));
    let want: Vec<u8> = unhex(abc).iter().map(|b| b ^ 0xA5).collect();
    assert_eq!(mixed.as_slice(), want.as_slice());
}

#[test]
fn rejects_bad_inputs() {
    let t = [0u8; 16];
    let base = BlockHashRequest {
        algo: HashAlgo::Ahash64,
        tokens: &t,
        token_width: 4,
        block_size: 2,
        parent: None,
        seed: None,
        salt: None,
        overrides: &[],
    };
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            block_size: 0,
            ..base
        }),
        Err(HashError::ZeroBlockSize)
    );
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            token_width: 2,
            ..base
        }),
        Err(HashError::UnsupportedTokenWidth(2))
    );
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            tokens: &t[..15],
            ..base
        }),
        Err(HashError::RaggedTokens { len: 15, width: 4 })
    );
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            seed: Some(&t[..16]),
            ..base
        }),
        Err(HashError::BadSeedLen(16))
    );
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            parent: Some(&t[..7]),
            ..base
        }),
        Err(HashError::BadParentLen {
            algo: HashAlgo::Ahash64,
            len: 7
        })
    );
    assert_eq!(
        hash_blocks(&BlockHashRequest {
            algo: HashAlgo::Sha256,
            parent: Some(&t[..8]),
            ..base
        }),
        Err(HashError::BadParentLen {
            algo: HashAlgo::Sha256,
            len: 8
        })
    );
}

#[test]
fn partial_block_is_dropped() {
    let t = Rng(3).tokens(10);
    assert_eq!(run(HashAlgo::Sha256, &t, 4, None, None, None).len(), 2);
    assert!(run(HashAlgo::Sha256, &t[..12], 4, None, None, None).is_empty());
}
