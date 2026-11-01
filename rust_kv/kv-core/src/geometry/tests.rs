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
//! MAX's `jenga/test_compute_jenga_ratios.py`, ported case for case, and the
//! `effective_max_seq_length` cases of `jenga/test_jenga_block_manager.py`.
//!
//! The cases use the real cache geometry of models MAX serves, since the
//! interesting behavior is entirely in how their page sizes divide each other.
//! The vision-tower cases are hypothetical -- an encoder attends over a fixed
//! image and pages nothing today -- and price what admitting a third,
//! unrelated geometry into the shared pool would cost.
//!
//! Python's `test_padded_invalid_arguments` also rejects `cache_sizes` and
//! `stride_alignments` naming different caches; a [`CachePage`] carries both,
//! so that case cannot be written here. Python's environment-variable tests become
//! checks on [`SearchBounds`].

use super::{
    CachePage, GeometryError, JengaGeometry, LeafDemand, MaxSeqLen, SearchBounds,
    compute_jenga_ratios, lcm_of, max_seq_len_fitting_in_geometry, plan_jenga_geometry,
    pristine_pool_can_satisfy,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const PAGE_SIZE: u64 = 128;
const BF16: u64 = 2;

/// The bytes one page of an MHA cache holds, keys and values.
fn mha_page_bytes_of(num_layers: u64, n_kv_heads: u64, head_dim: u64, dtype_bytes: u64) -> u64 {
    2 * num_layers * PAGE_SIZE * n_kv_heads * head_dim * dtype_bytes
}

fn mha_page_bytes(num_layers: u64, n_kv_heads: u64, head_dim: u64) -> u64 {
    mha_page_bytes_of(num_layers, n_kv_heads, head_dim, BF16)
}

/// The bytes one page of an MLA cache holds, a single latent.
fn mla_page_bytes(num_layers: u64, latent_dim: u64) -> u64 {
    num_layers * PAGE_SIZE * latent_dim * BF16
}

/// The page of the SigLIP-shaped vision tower gemma4 and kimi both carry: 27
/// layers of 16 heads over a hidden size of 1152, so a head is 72 wide -- an
/// odd multiple of 8, which is what makes the tower expensive to pool with a
/// text cache.
fn siglip_vision_page_bytes() -> u64 {
    mha_page_bytes(27, 16, 1152 / 16)
}

/// The bytes one row of an MHA cache holds, the unit TMA indexes.
fn mha_row_bytes(n_kv_heads: u64, head_dim: u64) -> u64 {
    n_kv_heads * head_dim * BF16
}

fn pages(data_bytes: &[u64], stride_alignments: &[Option<u64>]) -> Vec<CachePage> {
    assert_eq!(data_bytes.len(), stride_alignments.len());
    data_bytes
        .iter()
        .zip(stride_alignments)
        .map(|(&data_bytes, &stride_alignment)| CachePage {
            data_bytes,
            stride_alignment,
        })
        .collect()
}

fn padded_bounds(max_padding_fraction: f64) -> SearchBounds {
    SearchBounds {
        max_padding_fraction,
        ..SearchBounds::default()
    }
}

/// Asserts every cache tiles the huge block exactly, and returns its size.
fn check_tiles_exactly(cache_sizes: &[u64], ratios: &[usize]) -> u64 {
    let huge_bytes: Vec<u64> = cache_sizes
        .iter()
        .zip(ratios)
        .map(|(&size, &ratio)| size * ratio as u64)
        .collect();
    assert!(
        huge_bytes.iter().all(|&bytes| bytes == huge_bytes[0]),
        "{huge_bytes:?}"
    );
    huge_bytes[0]
}

/// Asserts the allocator invariants every padded geometry owes the kernel.
///
/// A page must divide the huge block, or a huge block does not hold a whole
/// number of them. A page past its data must be a stride its kernels can
/// address, a multiple of its alignment, and a cache without one stays
/// packed.
fn check_aligned_and_tiling(
    geometry: &JengaGeometry,
    cache_sizes: &[u64],
    stride_alignments: &[Option<u64>],
) {
    for (cache, &page) in geometry.padded_sizes.iter().enumerate() {
        assert_eq!(geometry.huge_page_bytes % page, 0, "cache {cache}");
        assert_eq!(
            geometry.huge_page_bytes / page,
            geometry.ratios[cache] as u64,
            "cache {cache}"
        );
        if page != cache_sizes[cache] {
            let alignment = stride_alignments[cache].expect("a packed cache was padded");
            assert_eq!(page % alignment, 0, "cache {cache}");
        }
    }
}

#[test]
fn gemma4_31b() {
    // 60 layers on a 5:1 pattern: 50 sliding ones with 16 kv heads of 256, and
    // 10 global ones with 4 kv heads of 512.
    let cache_sizes = [mha_page_bytes(50, 16, 256), mha_page_bytes(10, 4, 512)];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // A global page is a tenth of a sliding one, so the sliding page is itself
    // the huge block: pairing these two caches wastes nothing and strands
    // nothing, since a sliding block never shares a huge block with a sibling.
    assert_eq!(geometry.ratios, [1, 10]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        100 * MIB
    );
    // 120 GiB holds 1228 huge blocks, one of which is the null block.
    assert_eq!(geometry.num_huge_blocks, 1228);
}

#[test]
fn gemma4_31b_with_quantized_scales() {
    // Quantizing the values to fp8 halves them and adds a scale per group of
    // 128 elements along head_dim, which is a separate cache of its own.
    let cache_sizes = [
        mha_page_bytes_of(50, 16, 256, 1),
        mha_page_bytes_of(50, 16, 256 / 128, BF16),
        mha_page_bytes_of(10, 4, 512, 1),
        mha_page_bytes_of(10, 4, 512 / 128, BF16),
    ];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // Scale pages are 64x smaller than their values, so they set the ratios
    // without enlarging the huge block: it stays the largest page, as above.
    assert_eq!(geometry.ratios, [1, 64, 10, 640]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        50 * MIB
    );
}

#[test]
fn gemma4_31b_with_vision_encoder() {
    let cache_sizes = [
        mha_page_bytes(50, 16, 256),
        mha_page_bytes(10, 4, 512),
        siglip_vision_page_bytes(),
    ];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // The tower's 15.2 MiB page carries 27 * 9 = 243 as its odd part, and the
    // text pages carry 25, so the lcm multiplies both in: a 23.7 GiB huge block
    // to pool caches whose pages are 100 MiB, 10 MiB and 15 MiB.
    assert_eq!(geometry.ratios, [243, 2430, 1600]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        24300 * MIB
    );
    // A 120 GiB budget buys five huge blocks, four of them allocatable, so the
    // pool can hand the text caches at most four sliding-sized chunks of
    // memory before it is empty. Sharing bytes with the tower this way is not
    // viable.
    assert_eq!(geometry.num_huge_blocks, 5);
}

#[test]
fn kimi_k2_5_mla_with_vision_encoder() {
    let cache_sizes = [mla_page_bytes(61, 512 + 64), siglip_vision_page_bytes()];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // MLA fares better than gemma4 against the same tower: both pages already
    // carry a factor of 9, so the lcm only picks up 61 and 27, landing on
    // 926 MiB -- not even a whole number of MiB, but exact in bytes.
    assert_eq!(geometry.ratios, [108, 61]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        971_440_128
    );
    assert_eq!(geometry.num_huge_blocks, 132);
}

#[test]
fn kimi_k2_5_mla_with_drafter_and_vision_encoder() {
    let cache_sizes = [
        mla_page_bytes(61, 512 + 64),
        mha_page_bytes(1, 64, 128),
        siglip_vision_page_bytes(),
    ];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // Three geometries with nothing in common: the drafter forces the 2^22 of
    // its power-of-two page, MLA forces 61, and the tower forces 3^5, giving a
    // 58 GiB huge block. The budget holds two of them, one of which is the
    // null block, so the pool can serve exactly one huge block's worth of
    // requests.
    assert_eq!(geometry.ratios, [6912, 14823, 3904]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        59292 * MIB
    );
    assert_eq!(geometry.num_huge_blocks, 2);
}

#[test]
fn kimi_k2_5_mla_with_eagle3_drafter() {
    // An MLA target of 61 layers over a 512 + 64 latent, drafted by a 1-layer
    // MHA eagle3 head with 64 kv heads of 128.
    let cache_sizes = [mla_page_bytes(61, 512 + 64), mha_page_bytes(1, 64, 128)];
    let geometry = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();

    // 61 layers over a latent of 64 * 9 leaves 549 = 9 * 61 odd, while the
    // drafter's page is a power of two, so the lcm keeps both odd parts: a
    // 2.1 GiB huge block for pages of 8.6 MiB and 4 MiB. The geometry is exact
    // but coarse -- a 120 GiB budget holds only 55 of them, 54 after the null
    // block, and one live draft block strands a whole one.
    assert_eq!(geometry.ratios, [256, 549]);
    assert_eq!(
        check_tiles_exactly(&cache_sizes, &geometry.ratios),
        2196 * MIB
    );
    assert_eq!(geometry.num_huge_blocks, 55);
}

#[test]
fn kimi_k2_5_mla_alone() {
    // A single cache needs no common multiple, so its page is the huge block
    // and the budget divides evenly.
    let page = mla_page_bytes(61, 512 + 64);
    let geometry = compute_jenga_ratios(120 * GIB, &[page], true).unwrap();

    assert_eq!(geometry.ratios, [1]);
    assert_eq!(geometry.num_huge_blocks as u64, 120 * GIB / page);
}

#[test]
fn identical_pages_share_a_huge_block() {
    // Two caches of the same geometry -- a draft model mirroring its target,
    // say -- tile the huge block one page each, so neither constrains the
    // other and the pool splits between them purely by demand.
    let page = mha_page_bytes(24, 8, 128);
    let geometry = compute_jenga_ratios(8 * GIB, &[page, page], true).unwrap();

    assert_eq!(geometry.ratios, [1, 1]);
    assert_eq!(geometry.num_huge_blocks as u64, 8 * GIB / page);
}

#[test]
fn budget_that_holds_one_huge_block_is_rejected() {
    let page = mha_page_bytes(50, 16, 256);

    // The single huge block it holds goes to the null block, leaving nothing
    // to serve requests with. One more page buys the smallest usable pool.
    let error = compute_jenga_ratios(page, &[page], true).unwrap_err();
    assert!(
        error.to_string().contains("null page every cache shares"),
        "{error}"
    );
    assert_eq!(
        compute_jenga_ratios(2 * page, &[page], true).unwrap(),
        JengaGeometry {
            num_huge_blocks: 2,
            huge_page_bytes: page,
            ratios: vec![1],
            padded_sizes: vec![page],
        }
    );
}

#[test]
fn one_huge_block_suffices_without_a_null_block() {
    let page = mha_page_bytes(50, 16, 256);

    // The Rust host pool hands out every huge block it is given, so a caller
    // that reserves no null block can use the budget the null block would have
    // taken -- but an empty budget is still no pool.
    let geometry = compute_jenga_ratios(page, &[page], false).unwrap();
    assert_eq!(
        (geometry.num_huge_blocks, geometry.huge_page_bytes),
        (1, page)
    );
    assert_eq!(geometry.ratios, [1]);
    let error = compute_jenga_ratios(page / 2, &[page], false).unwrap_err();
    assert!(
        error.to_string().contains("at least one of them"),
        "{error}"
    );
}

#[test]
fn budget_smaller_than_the_lcm_is_rejected() {
    // Neither page alone is large, but their lcm is, which is the failure mode
    // worth naming in the error: sizing looks fine per cache and still fails.
    let cache_sizes = [mla_page_bytes(61, 512 + 64), mha_page_bytes(1, 64, 128)];
    assert!(cache_sizes.iter().all(|&size| size < GIB));

    let error = compute_jenga_ratios(GIB, &cache_sizes, true)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("least common multiple of the page sizes"),
        "{error}"
    );

    // Since a budget that looks generous next to either page can still be too
    // small, the error has to name the huge block and the minimum it implies.
    assert!(error.contains("so it takes 2.14 GiB"), "{error}");
    assert!(
        error.contains("at least two of them -- 4.29 GiB"),
        "{error}"
    );
}

#[test]
fn invalid_arguments() {
    let page = mha_page_bytes(50, 16, 256);
    let error = compute_jenga_ratios(120 * GIB, &[], true).unwrap_err();
    assert_eq!(error, GeometryError::NoCaches);
    assert!(error.to_string().contains("cache_sizes must be non-empty"));

    let error = compute_jenga_ratios(120 * GIB, &[page, 0], true).unwrap_err();
    assert_eq!(error, GeometryError::ZeroPageSize { cache: 1 });
    assert!(error.to_string().contains("cache_sizes must be positive"));

    let error = compute_jenga_ratios(0, &[page], true).unwrap_err();
    assert_eq!(error, GeometryError::ZeroBudget);
    assert!(
        error
            .to_string()
            .contains("available_bytes must be positive")
    );
}

#[test]
fn padding_is_not_spent_when_pages_already_divide() {
    // gemma4's global page is exactly a tenth of its sliding one, so exact
    // tiling is already cheap and the search must not pad to beat it, even
    // with both caches allowed to.
    let cache_sizes = [mha_page_bytes(50, 16, 256), mha_page_bytes(10, 4, 512)];
    let alignments = [Some(mha_row_bytes(16, 256)), Some(mha_row_bytes(4, 512))];
    let geometry = plan_jenga_geometry(
        120 * GIB,
        &pages(&cache_sizes, &alignments),
        true,
        &SearchBounds::default(),
    )
    .unwrap();

    // Identical to the exact path, down to the block count.
    assert_eq!(
        geometry,
        compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap()
    );
    for (cache, &size) in cache_sizes.iter().enumerate() {
        assert_eq!(geometry.padding_fraction(cache, size), 0.0);
    }
    check_aligned_and_tiling(&geometry, &cache_sizes, &alignments);
}

#[test]
fn vision_encoder_stops_exploding_the_huge_block() {
    // The case the exact path cannot serve: the tower's 243 = 27 * 9 odd part
    // against the text caches' 25 forces a 23.7 GiB huge block, of which a
    // 120 GiB budget holds five -- four allocatable. Padding buys it back.
    let cache_sizes = [
        mha_page_bytes(50, 16, 256),
        mha_page_bytes(10, 4, 512),
        siglip_vision_page_bytes(),
    ];
    let alignments = [
        Some(mha_row_bytes(16, 256)),
        Some(mha_row_bytes(4, 512)),
        Some(mha_row_bytes(16, 1152 / 16)),
    ];
    let exact = compute_jenga_ratios(120 * GIB, &cache_sizes, true).unwrap();
    let geometry = plan_jenga_geometry(
        120 * GIB,
        &pages(&cache_sizes, &alignments),
        true,
        &padded_bounds(0.25),
    )
    .unwrap();

    assert_eq!(exact.huge_page_bytes, 24300 * MIB);
    // Three orders of magnitude off the exact block, and the block count goes
    // from unusable to a real pool.
    assert!(geometry.huge_page_bytes < 128 * MIB);
    assert!(geometry.num_huge_blocks > 1000);
    check_aligned_and_tiling(&geometry, &cache_sizes, &alignments);
    // Nothing pays more than the cap for it.
    for (cache, &size) in cache_sizes.iter().enumerate() {
        assert!(geometry.padding_fraction(cache, size) <= 0.25);
    }
}

#[test]
fn v4_pro_seven_leaf_geometry() {
    // DeepSeek-V4-Pro is the case that motivated padding: seven cache kinds
    // whose layer counts (30, 31, 61) are pairwise coprime, so every one of
    // those primes survives into the exact lcm and none of them cancel.
    let cache_sizes = [
        61 * 64 * 576,
        30 * 64 * 576,
        31 * 2 * 576,
        30 * 64 * 68,
        30 * 4 * 8192,
        31 * 8 * 4096,
        30 * 4 * 2048,
    ];
    let alignments = [576, 576, 576, 68, 8192, 4096, 2048].map(Some);
    // Exact tiling wants a quarter of a terabyte for one huge block, so the
    // exact path cannot even return a geometry here -- it rejects the budget.
    assert!(lcm_of(cache_sizes).unwrap() > 200 * GIB);
    let error = compute_jenga_ratios(100 * GIB, &cache_sizes, true).unwrap_err();
    assert!(error.to_string().contains("too small to build a pool"));

    let geometry = plan_jenga_geometry(
        100 * GIB,
        &pages(&cache_sizes, &alignments),
        true,
        &padded_bounds(0.25),
    )
    .unwrap();
    assert!(geometry.huge_page_bytes <= 128 * MIB);
    assert!(geometry.num_huge_blocks > 2000);
    check_aligned_and_tiling(&geometry, &cache_sizes, &alignments);
}

/// DeepSeek-V4.1-Flash's packed record leaves and its index keys, which take
/// a 16-byte stride.
fn dsv4_1_flash_pages() -> (Vec<u64>, Vec<Option<u64>>) {
    (
        vec![
            43 * 128 * 528,
            3 * 64 * 288,
            128 * 288,
            3 * 64 * 68,
            128 * 68,
        ],
        vec![None, None, None, Some(16), Some(16)],
    )
}

#[test]
fn only_caches_with_a_stride_alignment_are_padded() {
    // The index keys' 17 cannot tile the packed leaves' 2^12 * 3^3 * 11 * 43
    // base, which exact tiling would multiply by 17 into an 848 MiB block.
    // Padding them to 2^9 * 27 and 2^10 * 9 keeps the base itself.
    let (cache_sizes, alignments) = dsv4_1_flash_pages();
    let geometry = plan_jenga_geometry(
        25 * GIB,
        &pages(&cache_sizes, &alignments),
        true,
        &SearchBounds::default(),
    )
    .unwrap();

    assert_eq!(geometry.huge_page_bytes, 52_310_016);
    assert_eq!(
        geometry.padded_sizes,
        [
            cache_sizes[0],
            cache_sizes[1],
            cache_sizes[2],
            13_824,
            9_216
        ]
    );
    check_aligned_and_tiling(&geometry, &cache_sizes, &alignments);

    // With nothing opted in, nothing pads, and the exact block is all there
    // is even past the ceiling.
    let packed = [None; 5];
    let geometry = plan_jenga_geometry(
        25 * GIB,
        &pages(&cache_sizes, &packed),
        true,
        &padded_bounds(1.0),
    )
    .unwrap();
    assert_eq!(geometry.padded_sizes, cache_sizes);
    assert_eq!(
        Some(geometry.huge_page_bytes),
        lcm_of(cache_sizes.iter().copied())
    );
}

#[test]
fn padded_invalid_arguments() {
    let page = mha_page_bytes(50, 16, 256);
    let row = Some(mha_row_bytes(16, 256));
    let bounds = SearchBounds::default();
    let plan = |available_bytes, pages: &[CachePage]| {
        plan_jenga_geometry(available_bytes, pages, true, &bounds)
            .unwrap_err()
            .to_string()
    };

    assert!(plan(120 * GIB, &[]).contains("cache_sizes must be non-empty"));
    assert!(
        plan(120 * GIB, &pages(&[page], &[Some(0)])).contains("stride_alignments must be positive")
    );
    assert!(plan(0, &pages(&[page], &[row])).contains("available_bytes must be positive"));
    assert!(plan(page, &pages(&[page], &[row])).contains("too small to build a pool"));
}

/// One request's Gated DeltaNet state leaves: page bytes, then row bytes.
///
/// A state page is one whole state -- the layer count times a row -- so an
/// awkward layer count survives into the huge block whole.
fn gated_delta_state_bytes(num_layers: u64) -> (Vec<u64>, Vec<u64>) {
    let num_value_heads = 48;
    let conv_row = (128 * 16 * 2 + 128 * num_value_heads) * 3 * BF16;
    let recurrent_row = num_value_heads * 128 * 128 * BF16;
    (
        vec![num_layers * conv_row, num_layers * recurrent_row],
        vec![conv_row, recurrent_row],
    )
}

#[test]
fn a_hybrid_ssm_geometry_pads_only_what_opts_in() {
    let (mut cache_sizes, row_sizes) = gated_delta_state_bytes(48);
    cache_sizes.push(mha_page_bytes(61, 4, 144));
    // State leaves are addressed by packed row, so they never opt in.
    let packed = [None, None, None];
    let geometry = plan_jenga_geometry(
        120 * GIB,
        &pages(&cache_sizes, &packed),
        true,
        &padded_bounds(0.25),
    )
    .unwrap();
    assert_eq!(geometry.padded_sizes, cache_sizes);

    // Opting every leaf in pads them -- which is exactly what made the
    // buffers strided -- and buys a much finer allocation quantum.
    let aligned = [
        Some(row_sizes[0]),
        Some(row_sizes[1]),
        Some(mha_row_bytes(4, 144)),
    ];
    let padded = plan_jenga_geometry(
        120 * GIB,
        &pages(&cache_sizes, &aligned),
        true,
        &padded_bounds(0.25),
    )
    .unwrap();
    assert_ne!(padded.padded_sizes, cache_sizes);
    assert!(padded.num_huge_blocks > geometry.num_huge_blocks);
    check_aligned_and_tiling(&padded, &cache_sizes, &aligned);
}

#[test]
fn search_bounds_default_to_the_python_environment_defaults() {
    // MODULAR_KV_JENGA_MAX_PADDING_FRACTION, _MAX_HUGE_PAGE_BYTES and
    // _MAX_TILING_RATIO, unset.
    assert_eq!(
        SearchBounds::default(),
        SearchBounds {
            max_padding_fraction: 0.1,
            max_huge_page_bytes: 128 * MIB,
            max_tiling_ratio: 256,
        }
    );
}

#[test]
fn a_lower_huge_page_ceiling_bounds_the_padded_geometry() {
    // The v4-pro leaves are all under 3 MiB, so a 16 MiB ceiling still leaves
    // the search room, and it must not return a block above it.
    let cache_sizes = [61 * 64 * 576, 31 * 2 * 576, 30 * 64 * 68, 30 * 4 * 8192];
    let alignments = [576, 576, 68, 8192].map(Some);
    let bounds = SearchBounds {
        max_padding_fraction: 0.25,
        max_huge_page_bytes: 16 * MIB,
        ..SearchBounds::default()
    };
    let geometry =
        plan_jenga_geometry(100 * GIB, &pages(&cache_sizes, &alignments), true, &bounds).unwrap();
    assert!(geometry.huge_page_bytes <= 16 * MIB);
    check_aligned_and_tiling(&geometry, &cache_sizes, &alignments);
}

#[test]
fn a_least_common_multiple_past_u64_is_larger_than_any_budget() {
    // Five primes near 2^20: Python's exact lcm is a 100-bit integer, so the
    // exact path rejects every budget, while padding still finds a geometry.
    let primes = [1_048_573, 1_048_571, 1_048_559, 1_048_549, 1_048_517];
    let cache_sizes = primes.map(|prime| prime * 64);
    assert_eq!(lcm_of(cache_sizes), None);

    let error = compute_jenga_ratios(u64::MAX, &cache_sizes, true).unwrap_err();
    assert!(
        matches!(
            error,
            GeometryError::BudgetBelowExactTiling {
                huge_page_bytes: None,
                ..
            }
        ),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("so it takes more than"),
        "{error}"
    );

    let geometry = plan_jenga_geometry(
        120 * GIB,
        &pages(&cache_sizes, &[Some(64); 5]),
        true,
        &padded_bounds(0.25),
    )
    .unwrap();
    assert!(geometry.huge_page_bytes <= 128 * MIB);
}

#[test]
fn pristine_pool_charges_every_cache_against_the_same_huge_blocks() {
    // Three blocks of a ratio-4 cache claim one huge block, five of a ratio-2
    // cache claim three.
    assert!(pristine_pool_can_satisfy(4, &[4, 2], &[3, 5]));
    assert!(!pristine_pool_can_satisfy(3, &[4, 2], &[3, 5]));
    assert!(pristine_pool_can_satisfy(0, &[4, 2], &[0, 0]));
}

#[test]
fn pristine_pool_does_not_wrap_an_overcommitted_charge_into_a_fit() {
    // The charges total `usize::MAX + 1` huge blocks, one more than the pool
    // holds, so a sum that saturated at `usize::MAX` would call this a fit.
    assert!(!pristine_pool_can_satisfy(
        usize::MAX,
        &[1, 1],
        &[usize::MAX, 1]
    ));
    assert!(pristine_pool_can_satisfy(
        usize::MAX,
        &[1, 1],
        &[usize::MAX, 0]
    ));
}

fn full() -> LeafDemand {
    LeafDemand::Full
}

fn sliding(window_size: usize, page_size: usize) -> LeafDemand {
    LeafDemand::SlidingWindow {
        window_size,
        page_size,
    }
}

#[test]
fn effective_max_seq_length_single_full_leaf() {
    // One leaf gets every allocatable huge block, so capacity grows linearly.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[full()], 4, 4, &[1], false),
        MaxSeqLen::Longest(4 * 4)
    );
}

#[test]
fn effective_max_seq_length_balances_across_full_leaves() {
    // 3 allocatable huge blocks: a=1,b=0 -> a=1,b=1 -> a=2,b=1.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[full(), full()], 1, 3, &[1, 1], false),
        MaxSeqLen::Longest(1)
    );
}

#[test]
fn effective_max_seq_length_weighs_by_ratio() {
    // A higher ratio grows a leaf's capacity faster per huge block.
    // 3 allocatable huge blocks: a=1,b=0 -> a=1,b=3 -> a=2,b=3.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[full(), full()], 1, 3, &[1, 3], false),
        MaxSeqLen::Longest(2)
    );
}

#[test]
fn effective_max_seq_length_sliding_window_saturates_to_none() {
    // Once every leaf's window is fully covered, more blocks add no more
    // capacity, so there is no finite bound on how long a sequence it serves.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[sliding(3, 1)], 1, 9, &[1], false),
        MaxSeqLen::Unbounded
    );
}

#[test]
fn effective_max_seq_length_sliding_window_below_saturation() {
    // 2 allocatable huge blocks, short of the 3 needed to saturate the window.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[sliding(3, 1)], 1, 2, &[1], false),
        MaxSeqLen::Longest(2)
    );
}

#[test]
fn effective_max_seq_length_full_leaf_is_the_bottleneck_once_swa_caps() {
    // 5 allocatable huge blocks: full=1,sliding=0 -> full=1,sliding=1
    // -> full=2,sliding=1 -> full=2,sliding capped -> full=3.
    assert_eq!(
        max_seq_len_fitting_in_geometry(&[full(), sliding(2, 1)], 1, 5, &[1, 1], false),
        MaxSeqLen::Longest(3)
    );
}

#[test]
fn a_state_that_overflows_the_pool_fits_nothing() {
    // With prefix caching each state draws a live and a checkpoint block
    // before any token: four huge blocks at a ratio of 1, for a pool of one.
    let states = [LeafDemand::Recurrent, LeafDemand::Recurrent];
    assert_eq!(
        max_seq_len_fitting_in_geometry(&states, 1, 1, &[1, 1], true),
        MaxSeqLen::NothingFits
    );
    assert_eq!(
        max_seq_len_fitting_in_geometry(&states, 1, 2, &[1, 1], false),
        MaxSeqLen::Unbounded
    );
}
