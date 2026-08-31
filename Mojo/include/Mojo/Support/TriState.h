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
//
// This file defines TriState, a three-valued (Kleene) logic result: provably
// true, provably false, or not statically decidable. It provides the small
// three-valued algebra (Kleene AND/OR and an AND-fold) in one place.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_TRISTATE_H
#define KGEN_SUPPORT_TRISTATE_H

#include <cstdint>
#include <optional>

namespace M::KGEN {

/// A three-valued (Kleene) logic result: a predicate is provably true (`yes`),
/// provably false (`no`), or not statically decidable (`unknown`).
class [[nodiscard]] TriState {
  enum class State : uint8_t { False, Unknown, True } state;
  constexpr explicit TriState(State s) : state(s) {}

public:
  static constexpr TriState yes() { return TriState(State::True); }
  static constexpr TriState no() { return TriState(State::False); }
  static constexpr TriState unknown() { return TriState(State::Unknown); }
  static constexpr TriState fromBool(bool b) { return b ? yes() : no(); }

  constexpr bool isTrue() const { return state == State::True; }
  constexpr bool isFalse() const { return state == State::False; }
  constexpr bool isUnknown() const { return state == State::Unknown; }
  /// True when the result is definitely known.
  constexpr bool isDefinite() const { return state != State::Unknown; }

  /// Returns std::nullopt when unknown, otherwise the definite boolean value.
  constexpr std::optional<bool> toOptionalBool() const {
    return isUnknown() ? std::nullopt : std::optional<bool>(isTrue());
  }

  /// Kleene AND: any `no` -> `no`; else any `unknown` -> `unknown`; else `yes`.
  constexpr TriState operator&(TriState o) const {
    if (isFalse() || o.isFalse())
      return no();
    if (isUnknown() || o.isUnknown())
      return unknown();
    return yes();
  }
  constexpr TriState &operator&=(TriState o) { return *this = *this & o; }

  /// Kleene OR: any `yes` -> `yes`; else any `unknown` -> `unknown`; else `no`.
  constexpr TriState operator|(TriState o) const {
    if (isTrue() || o.isTrue())
      return yes();
    if (isUnknown() || o.isUnknown())
      return unknown();
    return no();
  }
  constexpr TriState &operator|=(TriState o) { return *this = *this | o; }

  friend constexpr bool operator==(TriState a, TriState b) {
    return a.state == b.state;
  }
  friend constexpr bool operator!=(TriState a, TriState b) { return !(a == b); }
};

/// Fold a range of TriState values under Kleene AND ("all must hold"). An empty
/// range yields `yes()` (vacuous truth).
template <class Range>
constexpr TriState allTrue(Range &&r) {
  TriState acc = TriState::yes();
  for (TriState x : r) {
    acc &= x;
    // Short-circuit: once a definite `no` is seen the result can't change.
    if (acc.isFalse())
      break;
  }
  return acc;
}

} // namespace M::KGEN

#endif // KGEN_SUPPORT_TRISTATE_H
