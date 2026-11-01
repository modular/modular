//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Tests for debug allocation poison fill functionality.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DebugAllocPoison.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <system_error>

using namespace M;
using namespace M::GraphCompiler::Driver;

namespace {

class DebugAllocPoisonTest : public ::testing::Test {
protected:
  static ErrorOr<M::Driver::DeviceContextRef> makeGPU() {
    return M::Driver::DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, 0);
  }

  static M::Driver::DeviceContextRef makeCPU() {
    auto ctxOrErr = M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
    assert(!ctxOrErr.isError());
    return ctxOrErr.takeValue();
  }

  static void check16BitPattern(const void *data, size_t count,
                                uint16_t pattern) {
    const uint16_t *ptr = static_cast<const uint16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_EQ(ptr[i], pattern) << "Element " << i << " doesn't match pattern";
    }
  }
};

TEST_F(DebugAllocPoisonTest, BFloat16) {
  auto gpuOrErr = makeGPU();

  constexpr size_t numElements = 16;
  TensorSpec spec({numElements}, DType::bf16);

  // Both host and device fills use the largest-finite bit pattern (e.g.
  // 0x7F7F for bf16); a non-NaN, non-Inf value chosen so the poison fill
  // doesn't conflict with the nan-check debug pass.
  const llvm::fltSemantics *sem = spec.getEltType().getFloatSemantics();
  llvm::APFloat largest = llvm::APFloat::getLargest(*sem);
  uint16_t expectedPattern =
      static_cast<uint16_t>(largest.bitcastToAPInt().getZExtValue());

  if (!gpuOrErr.isError()) {
    auto gpu = gpuOrErr.takeValue();

    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, gpuBuf,
                  allocateDeviceBuffer(gpu, spec, /*hostAlloc=*/false));

    // Create host memory for verification
    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, hostBuf,
                  allocateDeviceBuffer(gpu, spec, /*hostAlloc=*/true));

    // Copy from GPU to host and synchronize
    EXPECT_NO_ERR(gpuBuf->copyTo(*hostBuf));
    EXPECT_NO_ERR(gpu->synchronize());

    check16BitPattern(hostBuf->ptr(), numElements, expectedPattern);
  } else {
    auto cpu = makeCPU();

    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, hostBuf,
                  allocateDeviceBuffer(cpu, spec, /*hostAlloc=*/true));

    check16BitPattern(hostBuf->ptr(), numElements, expectedPattern);
  }
}

bool isPoisoned(const AllocationPoisonDecision &decision,
                const TensorSpec &spec) {
  std::optional<PoisonPattern> pattern = getPoisonPattern(spec);
  return decision.fill && pattern && decision.fill->bits == pattern->bits &&
         decision.fill->patternBytes == pattern->patternBytes;
}

bool isZeroed(const AllocationPoisonDecision &decision) {
  return decision.fill && decision.fill->bits == 0;
}

AllocationPoisonDecision decide(llvm::StringRef selection,
                                const TensorSpec &spec, uint64_t ordinal = 0) {
  return decidePoisonForTesting(selection, spec, ordinal);
}

TEST(DebugAllocPoisonSelectionTest, DefaultAndAllPoisonEverything) {
  TensorSpec spec({16}, DType::si64);
  EXPECT_TRUE(isPoisoned(decide("", spec), spec));
  EXPECT_TRUE(isPoisoned(decide("all", spec, 12345), spec));
}

TEST(DebugAllocPoisonSelectionTest, NoneZeroFillsEverything) {
  TensorSpec spec({16}, DType::si64);
  EXPECT_TRUE(isZeroed(decide("none", spec)));
}

TEST(DebugAllocPoisonSelectionTest, SyncOnlyWritesNothing) {
  // `none` zero-fills, which makes a genuine uninitialized read deterministic
  // and so cannot separate "the fill mattered" from "the synchronization
  // mattered". `sync-only` writes nothing and is what separates them.
  TensorSpec spec({16}, DType::si64);
  EXPECT_FALSE(decide("sync-only", spec).fill);
}

TEST(DebugAllocPoisonSelectionTest, UnselectedDTypeWithoutPatternIsZeroed) {
  // Leaving it alone would make the unselected arm of a bisection
  // non-deterministic wherever fp8 buffers are read before being written.
  TensorSpec spec({16}, DType::f8e4m3fn);
  ASSERT_FALSE(getPoisonPattern(spec));
  EXPECT_TRUE(isZeroed(decide("none", spec)));
  EXPECT_TRUE(isZeroed(decide("ord=5", spec, 0)));
  EXPECT_TRUE(isZeroed(decide("dtype=si64", spec)));
}

TEST(DebugAllocPoisonSelectionTest, SelectedDTypeWithoutPatternIsLeftAlone) {
  TensorSpec spec({16}, DType::f8e4m3fn);
  EXPECT_FALSE(decide("all", spec).fill);
}

TEST(DebugAllocPoisonSelectionTest, DtypeClauseSelectsByElementType) {
  TensorSpec si64Spec({16}, DType::si64);
  TensorSpec ui32Spec({16}, DType::ui32);
  TensorSpec f32Spec({16}, DType::f32);

  EXPECT_TRUE(isPoisoned(decide("dtype=si64", si64Spec), si64Spec));
  EXPECT_TRUE(isZeroed(decide("dtype=si64", ui32Spec)));
  EXPECT_TRUE(isPoisoned(decide("dtype=si64,ui32", ui32Spec), ui32Spec));
  EXPECT_TRUE(isZeroed(decide("dtype=si64,ui32", f32Spec)));
}

TEST(DebugAllocPoisonSelectionTest, UnknownDTypeDoesNotNarrow) {
  // `int64` is not how DType spells si64, and `invalid` parses as a DType that
  // no allocation has. Matching nothing would report a clean run, so the
  // whole clause is ignored instead.
  TensorSpec si64Spec({16}, DType::si64);
  TensorSpec f32Spec({16}, DType::f32);
  EXPECT_TRUE(isPoisoned(decide("dtype=int64", si64Spec), si64Spec));
  EXPECT_TRUE(isPoisoned(decide("dtype=si64,bogus", f32Spec), f32Spec));
  EXPECT_TRUE(isPoisoned(decide("dtype=invalid", si64Spec), si64Spec));
}

TEST(DebugAllocPoisonSelectionTest, OrdinalClauseIsInclusiveOnBothEnds) {
  TensorSpec spec({16}, DType::si64);
  EXPECT_TRUE(isZeroed(decide("ord=10-20", spec, 9)));
  EXPECT_TRUE(isPoisoned(decide("ord=10-20", spec, 10), spec));
  EXPECT_TRUE(isPoisoned(decide("ord=10-20", spec, 20), spec));
  EXPECT_TRUE(isZeroed(decide("ord=10-20", spec, 21)));
}

TEST(DebugAllocPoisonSelectionTest, SingleOrdinalSelectsOne) {
  TensorSpec spec({16}, DType::si64);
  EXPECT_TRUE(isZeroed(decide("ord=7", spec, 6)));
  EXPECT_TRUE(isPoisoned(decide("ord=7", spec, 7), spec));
  EXPECT_TRUE(isZeroed(decide("ord=7", spec, 8)));
}

TEST(DebugAllocPoisonSelectionTest, MalformedOrdinalRangeDoesNotNarrow) {
  // A range with one bad bound must not keep the other, and a reversed range
  // must not select nothing.
  TensorSpec spec({16}, DType::si64);
  for (const char *selection :
       {"ord=x-y", "ord=10-x", "ord=x-10", "ord=-5", "ord=20-10", "ord="}) {
    EXPECT_TRUE(isPoisoned(decide(selection, spec, 0), spec)) << selection;
    EXPECT_TRUE(isPoisoned(decide(selection, spec, 100), spec)) << selection;
  }
}

TEST(DebugAllocPoisonSelectionTest, ClausesConjoin) {
  TensorSpec si64Spec({16}, DType::si64);
  TensorSpec ui32Spec({16}, DType::ui32);

  // Both clauses must hold, which is what makes a bisection over ordinals
  // within one dtype expressible.
  EXPECT_TRUE(
      isPoisoned(decide("dtype=si64&ord=0-100", si64Spec, 50), si64Spec));
  EXPECT_TRUE(isZeroed(decide("dtype=si64&ord=0-100", si64Spec, 101)));
  EXPECT_TRUE(isZeroed(decide("dtype=si64&ord=0-100", ui32Spec, 50)));
}

TEST(DebugAllocPoisonSelectionTest, SemicolonDoesNotSeparateClauses) {
  // `modular.cfg` cuts a value at `;`, so `;` cannot be the separator. Used
  // anyway, the whole thing is one malformed `dtype=` clause and must not
  // narrow.
  TensorSpec ui32Spec({16}, DType::ui32);
  EXPECT_TRUE(
      isPoisoned(decide("dtype=si64;ord=0-100", ui32Spec, 101), ui32Spec));
}

TEST(DebugAllocPoisonSelectionTest, UnknownClauseDoesNotNarrow) {
  TensorSpec spec({16}, DType::si64);
  EXPECT_TRUE(isPoisoned(decide("nonsense=1", spec, 7), spec));
}

TEST(DebugAllocPoisonSelectionTest, ModeOrRepeatedFilterInCompoundIsIgnored) {
  // Taken as conjunctions, each of these would select nothing.
  TensorSpec si64Spec({16}, DType::si64);
  EXPECT_TRUE(isPoisoned(decide("none&ord=5", si64Spec, 5), si64Spec));
  EXPECT_TRUE(
      isPoisoned(decide("ord=0-10&ord=100-200", si64Spec, 5), si64Spec));
  EXPECT_TRUE(isPoisoned(decide("dtype=si64&dtype=ui32", si64Spec), si64Spec));
}

TEST(DebugAllocPoisonSelectionTest, SelectionFileResetsOrdinalsOnEveryChange) {
  llvm::SmallString<128> path;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("poison-select", "txt", path));
  llvm::SmallString<128> tmpPath(path);
  tmpPath += ".tmp";
  // Replaced by rename, as the header documents: an in-place rewrite of the
  // same content can land in the same modification-time tick and go unseen.
  auto write = [&](llvm::StringRef spec) {
    {
      std::error_code ec;
      llvm::raw_fd_ostream out(tmpPath, ec);
      ASSERT_FALSE(ec);
      out << spec;
    }
    ASSERT_FALSE(llvm::sys::fs::rename(tmpPath, path));
  };

  TensorSpec spec({16}, DType::si64);
  write("ord=0-1");
  resetPoisonSelectionForTesting(path);

  EXPECT_TRUE(isPoisoned(decidePoison(spec, /*hostAlloc=*/true), spec));
  EXPECT_TRUE(isPoisoned(decidePoison(spec, /*hostAlloc=*/true), spec));
  EXPECT_TRUE(isZeroed(decidePoison(spec, /*hostAlloc=*/true)));

  // Rewriting the SAME selection must still reset, so that repeating a
  // measurement repeats its ordinals rather than continuing them.
  write("ord=0-1");
  AllocationPoisonDecision first = decidePoison(spec, /*hostAlloc=*/true);
  EXPECT_EQ(first.ordinal, 0u);
  EXPECT_TRUE(isPoisoned(first, spec));

  write("sync-only");
  AllocationPoisonDecision syncOnly = decidePoison(spec, /*hostAlloc=*/true);
  EXPECT_EQ(syncOnly.ordinal, 0u);
  EXPECT_FALSE(syncOnly.fill);

  resetPoisonSelectionForTesting("");
  llvm::sys::fs::remove(path);
}

TEST(DebugAllocPoisonSelectionTest, EmptyResetPathRestoresConfiguredSelection) {
  // The test runs with no selection configured, so that is `all`.
  resetPoisonSelectionForTesting("");
  TensorSpec spec({16}, DType::si64);
  AllocationPoisonDecision decision = decidePoison(spec, /*hostAlloc=*/true);
  EXPECT_EQ(decision.ordinal, 0u);
  EXPECT_TRUE(isPoisoned(decision, spec));
}

} // namespace
