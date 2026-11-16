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

#include "Mojo/KGENDialect/KGENParameters.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "gtest/gtest.h"

#include <cstdint>

using namespace M;
using namespace M::KGEN;

namespace {
using Analysis = ParameterCollector::Analysis;
using Info = Analysis::ParameterlessInfo;

const void *key(uintptr_t i) { return reinterpret_cast<const void *>(i << 4); }
} // namespace

TEST(ParameterCollectorCacheTest, LayerReadsThroughToBase) {
  Analysis base;
  base.try_emplace(key(1), Info{/*hasCtxEvalExpr=*/true, /*depth=*/2});
  Analysis layer = Analysis::makeThreadCache(base);

  const Info *info = layer.lookup(key(1));
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->hasCtxEvalExpr);
  EXPECT_EQ(info->requiredSignatureDepth, 2u);
  EXPECT_EQ(layer.lookup(key(2)), nullptr);
}

TEST(ParameterCollectorCacheTest, LayerInsertDoesNotChangeBase) {
  Analysis base;
  Analysis layer = Analysis::makeThreadCache(base);
  layer.try_emplace(key(1), Info{false, 1});

  EXPECT_NE(layer.lookup(key(1)), nullptr);
  EXPECT_EQ(base.lookup(key(1)), nullptr);
}

TEST(ParameterCollectorCacheTest, CopyOfLayerIsEmptyLayer) {
  // `failableParallelForEach` gives each worker a copy of the cache it is
  // passed, so a copy of an empty layer must also read through to the base.
  Analysis base;
  base.try_emplace(key(1), Info{false, 0});
  Analysis layer = Analysis::makeThreadCache(base);
  Analysis copy = layer;

  EXPECT_NE(copy.lookup(key(1)), nullptr);
  Analysis target;
  copy.mergeInto(target);
  EXPECT_EQ(target.lookup(key(1)), nullptr);
}

TEST(ParameterCollectorCacheTest, MergeIntoAddsOnlyLayerEntries) {
  Analysis base;
  base.try_emplace(key(1), Info{false, 0});
  Analysis layer = Analysis::makeThreadCache(base);
  layer.try_emplace(key(2), Info{false, 3});

  Analysis target;
  layer.mergeInto(target);
  EXPECT_EQ(target.lookup(key(1)), nullptr);
  const Info *info = target.lookup(key(2));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->requiredSignatureDepth, 3u);
}

// A signature mismatch would make `ThreadLocalCache` copy the cache silently.
static_assert(HasThreadCacheFactory<Analysis>);
