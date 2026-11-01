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
//! Which prefix of a hash chain a cache tree can reuse.
//!
//! The rules here are about attention shape alone. They take residency as a
//! predicate, so the same code answers for any tier: the device pools, or a KV
//! connector reporting what an external store holds (`GroupLoadPlan`'s
//! residency mask, in dkv-connector). A connector therefore needs no notion of
//! full attention, sliding window or SSM -- it reports presence, and this
//! decides what that presence is worth.
//!
//! Two shapes, and they behave differently under narrowing:
//!
//! * a **full-attention** group reads its whole history, so its hit is the run
//!   from the root and shortening an answer can never invalidate it;
//! * a **sliding-window** group reads only the last `blocks_in_window` blocks
//!   before wherever the sequence stops, so its hit is a SUFFIX run whose
//!   validity moves with the stopping point. Shortening can invalidate it.
//!
//! That second property is why [`longest_joint_prefix_hit`] iterates instead of
//! taking a minimum: narrowing for one group can invalidate another's window,
//! which narrows it again.
//!
//! MAX and Mach both run these rules, MAX through the `max._kv_core` extension,
//! so a hybrid-model bug is fixed once rather than twice (SERVOPT-1621). MAX
//! also has recurrent and scratch leaves, which [`LeafShape`] covers; Mach does
//! not have them yet and reaches the rules through [`longest_group_hit`].

/// The run of cached blocks from the root, which is what a full group reads.
///
/// `num_hashes` is how many blocks the request wants and `is_cached` whether
/// the block at an index is held. The result is downward-closed: a group
/// serving `n` also serves anything shorter.
#[must_use]
pub fn longest_full_attention_hit(num_hashes: usize, is_cached: impl Fn(usize) -> bool) -> usize {
    (0..num_hashes).take_while(|&idx| is_cached(idx)).count()
}

/// The deepest stopping point whose window this group holds complete.
///
/// Prefix-cache hits for a sliding window differ greatly from full attention.
/// The window size includes the query token: with the query token at
/// `idx=42` and a window size of 10, the query attends to tokens `idx=32`
/// through `idx=41`. Per token, for example:
///
/// ```text
/// [X]: Token is in Prefix Cache
///  . : Token is not in Prefix Cache
///  ^ : Eligible Prefix Cache hit
///
///   Tokens [A]  [B]   .   [D]  [E]  [F]   .    .   [I]  [J]  [K]  [L]  [M]
/// w_size=2  ^    ^         ^    ^    ^              ^    ^    ^    ^    ^
/// w_size=3  ^    ^              ^    ^                   ^    ^    ^    ^
/// w_size=4  ^    ^                   ^                        ^    ^    ^
/// w_size=5  ^    ^                                                 ^    ^
/// w_size=6  ^    ^                                                      ^
/// w_size=7  ^    ^
/// ```
///
/// As the window grows, fewer stopping points are eligible. Eligibility is a
/// consecutive run of `window_size - 1` tokens: `[DEF]` is a run of 3 tokens,
/// so `F` is a valid hit for `w_size=4` and below. A partial window is also a
/// hit when the run starts at the root: `[A]` and `[AB]` are hits for any
/// window size. Here the same rule is applied in whole blocks, with
/// `blocks_in_window` the window less the query token, rounded up to blocks.
///
/// Walks backwards counting a consecutive run, because the window sits at the
/// END of the candidate: the first place the run reaches `blocks_in_window` is
/// the deepest stopping point that works. A shallower one may not -- its window
/// covers different blocks -- so this cannot be found by shortening a
/// full-attention answer.
///
/// A run that survives to index 0 is a hit of just that run, with no window
/// check: nothing sits below the root to be missing.
///
/// `blocks_in_window` is at least 1 for every group that gets here. Zero is
/// `window_size == 1`, a query token attending to no history, which would make
/// every page a hit that attention never reads back; `KVCacheConfig::validate`
/// rejects that window and so does MAX's `KVCacheGroupId` (SERVOPT-1627).
/// Debug-asserted rather than reported: this sits on the per-request load path
/// and a zero has no answer to return, only a caller to fix.
#[must_use]
pub fn longest_sliding_window_hit(
    num_hashes: usize,
    blocks_in_window: usize,
    is_cached: impl Fn(usize) -> bool,
) -> usize {
    debug_assert!(
        blocks_in_window > 0,
        "a sliding-window group spans at least one block",
    );
    let mut run = 0usize;
    for idx in (0..num_hashes).rev() {
        if !is_cached(idx) {
            run = 0;
            continue;
        }
        run += 1;
        if run >= blocks_in_window {
            return idx + run;
        }
    }
    run
}

/// The longest prefix EVERY group can serve at once.
///
/// Each group answers under the run the others have already allowed, so the run
/// is settled once every group has accepted it in turn. A group asked again
/// about a prefix it just returned has to return that same length, or the loop
/// would walk the run to nothing.
///
/// Iterating is required rather than tidy. A minimum over the groups' answers
/// is wrong twice over: a windowed group's answer is not a depth that can be
/// compared with a full group's, and narrowing for one windowed group can
/// invalidate another's window, which narrows it again.
///
/// `group_hit` answers for group `k` how much of a candidate length it can
/// serve. Indexed rather than taking one closure per group, because every
/// caller holds its groups in a slice already and building a closure each would
/// allocate per request on the load path. Returns the agreed prefix length, or
/// `0` when the groups cannot agree on any.
///
/// An answer is clamped to the candidate rather than asserted equal to it: this
/// is `pub`, so the closure is not always one of this module's own rules, and a
/// larger answer would raise `candidate` while resetting `accepted` and need
/// not terminate. A `debug_assert` alone compiles out of the build that matters.
///
/// Each non-accepting turn drops the candidate by at least one and each turn
/// costs `O(candidate)`, so the bound is `O(num_hashes^2)` predicate
/// evaluations on the per-request load path. Real residency settles in a couple
/// of turns; the ceiling is only reachable with several windowed groups
/// disagreeing.
#[must_use]
pub fn longest_joint_prefix_hit(
    num_hashes: usize,
    num_groups: usize,
    mut group_hit: impl FnMut(usize, usize) -> usize,
) -> usize {
    if num_groups == 0 {
        return 0;
    }
    let mut candidate = num_hashes;
    let mut accepted = 0usize;
    let mut turn = 0usize;
    while candidate != 0 && accepted < num_groups {
        let num_hit_blocks = group_hit(turn, candidate).min(candidate);
        accepted = if num_hit_blocks == candidate {
            accepted + 1
        } else {
            1
        };
        candidate = num_hit_blocks;
        turn = (turn + 1) % num_groups;
    }
    candidate
}

/// How much of `candidate` a group of this shape serves: the rule above that
/// its `blocks_in_window` selects. [`LeafShape::longest_hit`] also covers
/// MAX's recurrent and scratch leaves.
#[must_use]
pub fn longest_group_hit(
    candidate: usize,
    blocks_in_window: Option<usize>,
    is_cached: impl Fn(usize) -> bool,
) -> usize {
    match blocks_in_window {
        None => longest_full_attention_hit(candidate, is_cached),
        Some(window) => longest_sliding_window_hit(candidate, window, is_cached),
    }
}

/// The deepest boundary a recurrent leaf holds a state for.
///
/// A recurrent state is published at one boundary rather than as a run of
/// blocks, so any resident boundary is a complete hit on its own and the
/// deepest one wins. A shallower candidate is answered from below it, which
/// keeps the joint rule's narrowing sound.
#[must_use]
pub fn longest_recurrent_hit(num_hashes: usize, is_cached: impl Fn(usize) -> bool) -> usize {
    (0..num_hashes)
        .rev()
        .find(|&idx| is_cached(idx))
        .map_or(0, |idx| idx + 1)
}

/// A leaf's attention shape, which selects the rule that reads its residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeafShape {
    /// Reads its whole history: [`longest_full_attention_hit`].
    Full,
    /// Reads only the last `blocks_in_window` blocks before the stopping point:
    /// [`longest_sliding_window_hit`]. The window is at least one block.
    SlidingWindow {
        /// The window in whole blocks, excluding the query token.
        blocks_in_window: usize,
    },
    /// Holds one state per published boundary: [`longest_recurrent_hit`].
    Recurrent,
    /// Never published, so it has no opinion on a hit and must not shorten
    /// what the leaves that do cache agree on.
    Scratch,
}

impl LeafShape {
    /// How much of `candidate` a leaf of this shape serves.
    #[must_use]
    pub fn longest_hit(self, candidate: usize, is_cached: impl Fn(usize) -> bool) -> usize {
        match self {
            // Reads its whole history, so the hit is the run from the root.
            Self::Full => longest_full_attention_hit(candidate, is_cached),
            // The hit is a SUFFIX run, so it cannot be found by shortening a
            // full-attention answer: a shallower stopping point's window
            // covers different blocks.
            Self::SlidingWindow { blocks_in_window } => {
                longest_sliding_window_hit(candidate, blocks_in_window, is_cached)
            }
            // A state is a single published boundary rather than a run, so the
            // deepest one that stands is the answer.
            Self::Recurrent => longest_recurrent_hit(candidate, is_cached),
            // Never published, so it has no opinion and must not shorten what
            // the leaves that do cache agree on.
            Self::Scratch => candidate,
        }
    }
}

/// How many blocks a group actually holds of a hit that deep.
///
/// The companion to the rules above, and not the same number: a windowed
/// group's hit can cover a whole prefix while the group holds only the window
/// at the end of it, because the slots below the window are never read. A full
/// group holds all of it.
///
/// Every group's blocks END at `num_hit_blocks`, so a group holding `n` of them
/// covers `hashes[num_hit_blocks - n .. num_hit_blocks]` and the
/// `num_hit_blocks - n` slots below are the null block. `blocks_in_window` is
/// the group's window in whole blocks, or `None` for a group that attends over
/// its whole history.
#[must_use]
pub fn blocks_held_of_hit(num_hit_blocks: usize, blocks_in_window: Option<usize>) -> usize {
    match blocks_in_window {
        None => num_hit_blocks,
        Some(window) => num_hit_blocks.min(window),
    }
}

/// One group's residency for a request, in ABSOLUTE page slots.
///
/// The adaptation between what a store answers (positions in what this group
/// asked for) and what the rules above take (absolute slots), so the two
/// engines' rules can be applied to a Mach round without restating the
/// slot-offset convention at each call site.
#[derive(Debug, Clone, Copy)]
pub struct GroupResidency<'a> {
    /// Absolute slot `resident[0]` describes.
    pub slot_offset: usize,
    /// Root-anchored local prefix already resident below `slot_offset`. Equal
    /// to `slot_offset` for an ordinary claim, and `0` for a null-headed window
    /// whose lower slots are genuinely absent. Values strictly between the two
    /// are not a shape any producer emits and are rejected in debug.
    pub local_prefix_pages: usize,
    /// This group's window in whole blocks, or `None` for full attention.
    pub blocks_in_window: Option<usize>,
    /// How many blocks this group was asked about, which bounds the slots it
    /// can speak for. Carried separately from `resident.len()` so a group that
    /// answered nothing at all still constrains the joint over its whole ask
    /// rather than dropping out of it.
    pub probed: usize,
    /// Blocks from `slot_offset` that the TRANSFER can move: the store's
    /// leading run (`GroupLoadPlan::matched_blocks`). Residency past a gap is
    /// visible to the rules but unreachable by `load_with_plan`, whose device
    /// ids pair positionally with a leading run, so this is what bounds a
    /// credit that a request will be staked on.
    pub loadable_blocks: usize,
    /// What the store answered, positionally over what this group asked for.
    /// May be shorter than `probed`; the missing tail reads absent.
    pub resident: &'a [bool],
}

impl GroupResidency<'_> {
    /// Residency at an ABSOLUTE slot. Below `slot_offset` the answer is the
    /// local prefix rather than the store's: those slots were never asked
    /// about, and a root-anchored claim already holds them.
    #[must_use]
    pub fn is_cached(&self, slot: usize) -> bool {
        if slot < self.slot_offset {
            // The branch is load-bearing on `local_prefix_pages` being one of
            // the two shapes: a value in between would report every
            // below-offset slot absent rather than only those past the prefix.
            // `joint_contribution` asserted this and no longer sits on the
            // load path, so the guard lives here now.
            debug_assert!(
                self.local_prefix_pages == 0 || self.local_prefix_pages == self.slot_offset,
                "local_prefix_pages {} is neither 0 nor slot_offset {}",
                self.local_prefix_pages,
                self.slot_offset,
            );
            return self.local_prefix_pages == self.slot_offset;
        }
        self.resident
            .get(slot - self.slot_offset)
            .copied()
            .unwrap_or(false)
    }

    /// One past the deepest slot this group can speak about.
    #[must_use]
    pub fn extent(&self) -> usize {
        self.slot_offset + self.probed
    }

    /// Deepest absolute prefix the transfer path can actually fill for this
    /// group. See [`loadable_blocks`][Self::loadable_blocks].
    #[must_use]
    pub fn loadable_depth(&self) -> usize {
        self.slot_offset + self.loadable_blocks
    }
}

/// The joint prefix these groups serve for one request, in ABSOLUTE page slots.
///
/// The whole reconciliation, in one place so a connector and a test fixture
/// cannot drift: each group answers under the run the others have allowed,
/// through the rule its own shape selects.
///
/// The candidate starts at the deepest slot any group was asked about, capped
/// by the shallowest depth any of them can actually FILL
/// ([`loadable_depth`][GroupResidency::loadable_depth]). Without that cap the
/// rules will happily credit a complete window sitting above a gap, which is a
/// true statement about residency and an unfillable one about transfer: the
/// caller stakes the request on the credit, `load_with_plan` moves the leading
/// run only, and the round fails and rolls back every member. Capping the
/// candidate rather than the answer keeps each group's rule evaluated only at
/// depths it could be asked to fill.
///
/// Ragged extents are admitted on purpose, where MAX's connector rejects them:
/// mach's groups carry per-group `slot_offset`s and ask different ranges, so
/// there is no single width to check against. A slot above a group's own extent
/// reads absent, which narrows the joint rather than widening it.
#[must_use]
pub fn longest_joint_residency_hit(groups: &[GroupResidency<'_>]) -> usize {
    let deepest_asked = groups.iter().map(GroupResidency::extent).max().unwrap_or(0);
    let fillable = groups
        .iter()
        .map(GroupResidency::loadable_depth)
        .min()
        .unwrap_or(0);
    longest_joint_prefix_hit(
        deepest_asked.min(fillable),
        groups.len(),
        |gi, candidate| {
            let g = &groups[gi];
            longest_group_hit(candidate, g.blocks_in_window, |slot| g.is_cached(slot))
        },
    )
}

/// A group's window in whole blocks, or `None` for a group attending over its
/// whole history.
///
/// `ceildiv(window_size - 1, page_size)`, which is the arithmetic MAX's
/// `KVCacheGroupId.blocks_in_window` runs for a windowed leaf (MAX returns a
/// `-1` sentinel for a full one where this returns `None`). The formula is kept
/// identical rather than simplified because these rules are shared with MAX,
/// and an off-by-one here moves the depth the two engines agree on.
#[must_use]
pub fn blocks_in_window(window: Option<usize>, page_size: usize) -> Option<usize> {
    window.map(|w| w.saturating_sub(1).div_ceil(page_size))
}

#[cfg(test)]
mod tests {
    use super::{
        GroupResidency, LeafShape, blocks_held_of_hit, blocks_in_window,
        longest_full_attention_hit, longest_group_hit, longest_joint_prefix_hit,
        longest_joint_residency_hit, longest_recurrent_hit, longest_sliding_window_hit,
    };

    /// The ported form of the joint rule: one callable per group, as
    /// `prefix_hit.py` spells it.
    fn joint_hit(num_hashes: usize, rules: &[&dyn Fn(usize) -> usize]) -> usize {
        longest_joint_prefix_hit(num_hashes, rules.len(), |gi, candidate| {
            rules[gi](candidate)
        })
    }

    /// A residency predicate over a positional mask.
    fn held(mask: &[bool]) -> impl Fn(usize) -> bool + '_ {
        move |idx| mask[idx]
    }

    fn full_rule(mask: &[bool]) -> impl Fn(usize) -> usize + '_ {
        move |candidate| longest_full_attention_hit(candidate, held(mask))
    }

    fn window_rule(mask: &[bool], blocks_in_window: usize) -> impl Fn(usize) -> usize + '_ {
        move |candidate| longest_sliding_window_hit(candidate, blocks_in_window, held(mask))
    }

    #[test]
    fn full_attention_stops_at_the_first_hole() {
        let mask = [true, true, true, false, true];
        assert_eq!(longest_full_attention_hit(5, held(&mask)), 3);
    }

    #[test]
    fn sliding_window_takes_the_deepest_complete_window() {
        // MAX's mask for this case, and deliberately one with TWO complete
        // windows -- maximal runs at 2..3 and 5..6, both long enough for a
        // window of 2 -- so returning the shallower one is a visible failure.
        // The window also sits past a gap here: slots 0 and 1 are missing and
        // are never read at this stopping point, so they cost nothing.
        let mask = [false, false, true, true, false, true, true];
        assert_eq!(longest_sliding_window_hit(7, 2, held(&mask)), 7);
    }

    #[test]
    fn sliding_window_falls_back_to_a_root_anchored_run() {
        // No run reaches the window, so all that is left is the run that
        // survives to the root, which needs no block under it.
        let mask = [true, true, false, true];
        assert_eq!(longest_sliding_window_hit(4, 3, held(&mask)), 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "spans at least one block")]
    fn a_window_spanning_no_whole_block_is_rejected() {
        // `window_size == 1`. It used to answer "everything is a hit"; the
        // window is refused at config validation now, so reaching the rule
        // with one is a caller bug (SERVOPT-1627).
        let _ = longest_sliding_window_hit(5, 0, held(&[false; 5]));
    }

    #[test]
    fn two_full_groups_agree_on_the_shorter_run() {
        let a = [true, true, true, false];
        let b = [true, true, false, false];
        assert_eq!(joint_hit(4, &[&full_rule(&a), &full_rule(&b)]), 2);
    }

    /// The case no minimum can reach. Full holds A..F; the windowed group holds
    /// `[C D]` and `[F G]` with window 2, and is missing E.
    ///
    /// ```text
    ///   stop at F -> window needs [E F] -> E missing
    ///   stop at E -> window needs [D E] -> E missing
    ///   stop at D -> window needs [C D] -> held
    /// ```
    ///
    /// E rules out BOTH of the first two candidates, so the answer is not
    /// reachable by shortening either group's own answer once.
    #[test]
    fn full_and_windowed_agree_below_both_their_answers() {
        let full = [true, true, true, true, true, true, false];
        let window = [false, false, true, true, false, true, true];
        assert_eq!(
            joint_hit(7, &[&full_rule(&full), &window_rule(&window, 2)]),
            4
        );
    }

    /// The windowed group holds a COMPLETE window, `[F G]`, but it ends past
    /// where the full group reaches. Capping the candidate at F leaves the
    /// windowed group holding only F: a run of 1 against a window of 2, and
    /// nothing shallower has two in a row either.
    #[test]
    fn a_window_the_full_group_cannot_reach_is_no_hit() {
        let full = [true, true, true, true, true, true, false];
        let window = [false, false, false, false, false, true, true];
        assert_eq!(
            joint_hit(7, &[&full_rule(&full), &window_rule(&window, 2)]),
            0
        );
    }

    /// Why the joint rule ITERATES rather than asking each group once. The
    /// windowed group answers 2 (B alone is a complete window of 1), which
    /// narrows the candidate to 2; the full group then answers 1, because its
    /// run from the root is just A. Re-asking the windowed group at 1 is what
    /// catches that B is gone from that candidate and A is not held: a single
    /// round-robin pass stops at 1 and claims a hit the KV cannot back.
    #[test]
    fn narrowing_reopens_a_group_already_asked() {
        let window = [false, true];
        let full = [true, false];
        assert_eq!(
            joint_hit(2, &[&window_rule(&window, 1), &full_rule(&full)]),
            0
        );
    }

    /// Two WINDOWED groups of different widths, which is the shape the run-and-
    /// min rule this replaced could not express at all:
    /// neither answer is a depth the other can be compared against, so a
    /// minimum is not merely imprecise here, it is unsound.
    ///
    /// Narrow (window 2) holds `[0 1 2]` and `[5 6]`; wide (window 3) holds
    /// `[0 1 2 3]`. Their own answers are 7 and 4, so a minimum claims 4. At 4
    /// the narrow group's window is `[2 3]` and it does not hold 3, so 4 is not
    /// servable. Only re-asking converges, on 3.
    #[test]
    fn two_windowed_groups_converge_below_the_minimum_of_their_answers() {
        let narrow = [true, true, true, false, false, true, true];
        let wide = [true, true, true, true, false, false, false];

        assert_eq!(longest_sliding_window_hit(7, 2, held(&narrow)), 7);
        assert_eq!(longest_sliding_window_hit(7, 3, held(&wide)), 4);

        let joint = joint_hit(7, &[&window_rule(&narrow, 2), &window_rule(&wide, 3)]);
        assert_eq!(
            joint, 3,
            "the minimum of the two answers (4) is not servable: the narrow \
             group's window at 4 needs block 3, which it does not hold",
        );
        // Both groups really do hold their window at the agreed depth.
        assert_eq!(longest_sliding_window_hit(joint, 2, held(&narrow)), joint);
        assert_eq!(longest_sliding_window_hit(joint, 3, held(&wide)), joint);
    }

    #[test]
    fn a_lone_windowed_group_needs_nothing_to_bound_it() {
        let window = [false, false, true, true];
        assert_eq!(joint_hit(4, &[&window_rule(&window, 2)]), 4);
    }

    #[test]
    fn no_groups_is_no_hit() {
        assert_eq!(joint_hit(5, &[]), 0);
    }

    /// MAX's own table for the same arithmetic
    /// (`test_blocks_in_window_excludes_the_query_token`). The `- 1` is what
    /// excludes the query token's own page, and it only shows up when
    /// `window ≡ 1 (mod page_size)` -- which no registry window reaches today,
    /// so without this the formula is protected by coincidence.
    #[test]
    fn blocks_in_window_excludes_the_query_token() {
        // Two is the shallowest window `KVCacheConfig::validate` admits.
        assert_eq!(blocks_in_window(Some(2), 128), Some(1));
        assert_eq!(blocks_in_window(Some(128), 128), Some(1));
        assert_eq!(blocks_in_window(Some(129), 128), Some(1));
        assert_eq!(blocks_in_window(Some(130), 128), Some(2));
        // Gemma 4's geometry, the one this PR is measured on.
        assert_eq!(blocks_in_window(Some(1024), 128), Some(8));
        // A full-attention group has no window at all, where MAX carries a
        // `-1` sentinel it translates at its own boundary.
        assert_eq!(blocks_in_window(None, 128), None);
    }

    /// A view that asked about `probed` blocks and can move `loadable_blocks`
    /// of them from `slot_offset`.
    fn view(
        slot_offset: usize,
        local_prefix_pages: usize,
        blocks_in_window: Option<usize>,
        probed: usize,
        loadable_blocks: usize,
        resident: &[bool],
    ) -> GroupResidency<'_> {
        GroupResidency {
            slot_offset,
            local_prefix_pages,
            blocks_in_window,
            probed,
            loadable_blocks,
            resident,
        }
    }

    /// The cap is this module's invariant, so this module proves it rather than
    /// leaving the proof one crate away. A windowed group holding a complete
    /// window above a gap is a true statement about residency and an unfillable
    /// one about transfer; the joint must not exceed any group's
    /// `loadable_depth`.
    #[test]
    fn a_joint_never_exceeds_what_a_group_can_move() {
        let full = [true; 19];
        let mut window = [false; 19];
        for slot in window.iter_mut().skip(11) {
            *slot = true;
        }
        // The rule does see the window; the leading run reaches nothing.
        assert_eq!(longest_sliding_window_hit(19, 8, held(&window)), 19);

        let groups = [
            view(0, 0, None, 19, 19, &full),
            view(0, 0, Some(8), 19, 0, &window),
        ];
        assert_eq!(longest_joint_residency_hit(&groups), 0);
    }

    /// Capping the CANDIDATE is not the same as capping the ANSWER, and this is
    /// the case that separates them. The windowed group can move one block from
    /// slot 3 and holds no complete window of 2 at or below that depth, so the
    /// only sound joint is 0. An answer-cap would run the rules unbounded and
    /// then clamp, keeping a number the windowed group never endorsed at the
    /// depth it was clamped to.
    #[test]
    fn the_cap_bounds_the_candidate_not_the_answer() {
        let full = [true; 12];
        let window = [true; 6];
        let groups = [
            view(0, 0, None, 12, 12, &full),
            view(3, 0, Some(2), 6, 1, &window),
        ];
        assert_eq!(longest_joint_residency_hit(&groups), 0);

        let answer_capped = longest_joint_prefix_hit(12, groups.len(), |gi, candidate| {
            let g = &groups[gi];
            longest_group_hit(candidate, g.blocks_in_window, |slot| g.is_cached(slot))
        })
        .min(
            groups
                .iter()
                .map(GroupResidency::loadable_depth)
                .min()
                .unwrap(),
        );
        assert_ne!(
            answer_capped, 0,
            "if these agreed, this test would not be pinning the choice",
        );
    }

    /// The below-offset branch decides whether a null head reads as present,
    /// and only a band whose held run is SHORTER than its window reaches it:
    /// the walk exhausts the band, resets, and steps down through the offset. A
    /// band whose run already fills its window returns before it gets there,
    /// which is why the partly-filled-band case does not cover this.
    #[test]
    fn a_null_head_reads_absent_below_its_offset() {
        let full = [true; 6];
        // Two held blocks at slots 4..6 against a window of four.
        let short_band = [true; 2];
        let null_headed = [
            view(0, 0, None, 6, 6, &full),
            view(4, 0, Some(4), 2, 2, &short_band),
        ];
        assert_eq!(
            longest_joint_residency_hit(&null_headed),
            0,
            "a null head must not read as resident: nothing fills the window",
        );

        // The same band whose head IS resident locally does serve it, which is
        // the other direction of the same branch.
        let local_head = [
            view(0, 0, None, 6, 6, &full),
            view(4, 4, Some(4), 2, 2, &short_band),
        ];
        assert_eq!(longest_joint_residency_hit(&local_head), 6);
    }

    #[test]
    fn a_full_group_holds_the_whole_hit() {
        assert_eq!(blocks_held_of_hit(4, None), 4);
    }

    #[test]
    fn a_windowed_group_holds_only_its_window() {
        assert_eq!(blocks_held_of_hit(4, Some(2)), 2);
    }

    #[test]
    fn a_window_wider_than_the_hit_holds_the_whole_hit() {
        assert_eq!(blocks_held_of_hit(2, Some(8)), 2);
    }

    #[test]
    fn a_recurrent_leaf_takes_its_deepest_state() {
        let mask = [true, false, true, false];
        assert_eq!(longest_recurrent_hit(4, held(&mask)), 3);
        // Narrowed below that state, it answers from the next one down.
        assert_eq!(longest_recurrent_hit(2, held(&mask)), 1);
        assert_eq!(longest_recurrent_hit(4, held(&[false; 4])), 0);
    }

    #[test]
    fn each_shape_selects_its_rule() {
        let mask = [true, false, true, true];
        let at = |shape: LeafShape| shape.longest_hit(mask.len(), held(&mask));
        assert_eq!(at(LeafShape::Full), 1);
        assert_eq!(
            at(LeafShape::SlidingWindow {
                blocks_in_window: 2
            }),
            4
        );
        assert_eq!(at(LeafShape::Recurrent), 4);
        assert_eq!(at(LeafShape::Scratch), 4);
        assert_eq!(LeafShape::Scratch.longest_hit(3, |_| false), 3);
    }

    #[test]
    fn a_scratch_leaf_never_narrows_the_joint() {
        let full = [true, true, false];
        let shapes = [LeafShape::Scratch, LeafShape::Full];
        let joint = longest_joint_prefix_hit(3, shapes.len(), |gi, candidate| {
            shapes[gi].longest_hit(candidate, held(&full))
        });
        assert_eq!(joint, 2);
    }

    #[test]
    fn a_recurrent_leaf_settles_with_a_window() {
        // The state at 4 is narrowed to 3 by the window, which drops the state
        // to the one at 2; the window holds [0, 1] complete, so both agree there.
        let state = [false, true, false, true];
        let window = [true, true, true, false];
        let shapes = [
            (LeafShape::Recurrent, &state),
            (
                LeafShape::SlidingWindow {
                    blocks_in_window: 2,
                },
                &window,
            ),
        ];
        let joint = longest_joint_prefix_hit(4, shapes.len(), |gi, candidate| {
            let (shape, mask) = shapes[gi];
            shape.longest_hit(candidate, held(mask))
        });
        assert_eq!(joint, 2);
    }
}
