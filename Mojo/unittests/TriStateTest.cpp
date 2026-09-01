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

#include "Mojo/include/Mojo/Support/TriState.h"
#include "gtest/gtest.h"

#include <array>

using M::KGEN::allTrue;
using M::KGEN::TriState;

namespace {

TEST(TriStateTest, factoriesAndPredicates) {
  EXPECT_TRUE(TriState::yes().isTrue());
  EXPECT_FALSE(TriState::yes().isFalse());
  EXPECT_FALSE(TriState::yes().isUnknown());
  EXPECT_TRUE(TriState::yes().isDefinite());

  EXPECT_TRUE(TriState::no().isFalse());
  EXPECT_FALSE(TriState::no().isTrue());
  EXPECT_FALSE(TriState::no().isUnknown());
  EXPECT_TRUE(TriState::no().isDefinite());

  EXPECT_TRUE(TriState::unknown().isUnknown());
  EXPECT_FALSE(TriState::unknown().isTrue());
  EXPECT_FALSE(TriState::unknown().isFalse());
  EXPECT_FALSE(TriState::unknown().isDefinite());
}

TEST(TriStateTest, fromBoolAndToOptionalBool) {
  EXPECT_TRUE(TriState::fromBool(true).isTrue());
  EXPECT_TRUE(TriState::fromBool(false).isFalse());

  EXPECT_EQ(TriState::yes().toOptionalBool(), std::optional<bool>(true));
  EXPECT_EQ(TriState::no().toOptionalBool(), std::optional<bool>(false));
  EXPECT_EQ(TriState::unknown().toOptionalBool(), std::nullopt);
}

TEST(TriStateTest, equality) {
  EXPECT_EQ(TriState::yes(), TriState::yes());
  EXPECT_NE(TriState::yes(), TriState::no());
  EXPECT_NE(TriState::yes(), TriState::unknown());
  EXPECT_NE(TriState::no(), TriState::unknown());
}

TEST(TriStateTest, kleeneAndTruthTable) {
  const TriState y = TriState::yes(), n = TriState::no(),
                 u = TriState::unknown();

  // Any `no` dominates.
  EXPECT_EQ(n & n, n);
  EXPECT_EQ(n & y, n);
  EXPECT_EQ(y & n, n);
  EXPECT_EQ(n & u, n);
  EXPECT_EQ(u & n, n);

  // Otherwise any `unknown` yields `unknown`.
  EXPECT_EQ(u & u, u);
  EXPECT_EQ(y & u, u);
  EXPECT_EQ(u & y, u);

  // All `yes` yields `yes`.
  EXPECT_EQ(y & y, y);
}

TEST(TriStateTest, kleeneOrTruthTable) {
  const TriState y = TriState::yes(), n = TriState::no(),
                 u = TriState::unknown();

  // Any `yes` dominates.
  EXPECT_EQ(y | y, y);
  EXPECT_EQ(y | n, y);
  EXPECT_EQ(n | y, y);
  EXPECT_EQ(y | u, y);
  EXPECT_EQ(u | y, y);

  // Otherwise any `unknown` yields `unknown`.
  EXPECT_EQ(u | u, u);
  EXPECT_EQ(n | u, u);
  EXPECT_EQ(u | n, u);

  // All `no` yields `no`.
  EXPECT_EQ(n | n, n);
}

TEST(TriStateTest, compoundAssignment) {
  TriState acc = TriState::yes();
  acc &= TriState::unknown();
  EXPECT_EQ(acc, TriState::unknown());
  acc &= TriState::no();
  EXPECT_EQ(acc, TriState::no());

  TriState orAcc = TriState::no();
  orAcc |= TriState::unknown();
  EXPECT_EQ(orAcc, TriState::unknown());
  orAcc |= TriState::yes();
  EXPECT_EQ(orAcc, TriState::yes());
}

TEST(TriStateTest, allTrueFold) {
  // Empty range is vacuously true.
  EXPECT_EQ(allTrue(std::array<TriState, 0>{}), TriState::yes());

  EXPECT_EQ(allTrue(std::array{TriState::yes(), TriState::yes()}),
            TriState::yes());
  EXPECT_EQ(allTrue(std::array{TriState::yes(), TriState::unknown()}),
            TriState::unknown());
  EXPECT_EQ(
      allTrue(std::array{TriState::yes(), TriState::unknown(), TriState::no()}),
      TriState::no());
}

// constexpr usability: the algebra must be evaluable at compile time.
TEST(TriStateTest, constexprEvaluation) {
  static_assert((TriState::yes() & TriState::unknown()).isUnknown());
  static_assert((TriState::no() | TriState::yes()).isTrue());
  static_assert(TriState::fromBool(true).isTrue());
  SUCCEED();
}

} // namespace
