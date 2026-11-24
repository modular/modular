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
// This file tests the DeviceGraphBuilder interface for explicit device graph
// construction.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "DeviceGraphTestUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstring>

using namespace M::Driver;
using M::ErrorOr;

namespace {

TEST(DeviceGraphBuilderTest, VecAddKernelNode) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Initialize host input buffers.
  llvm::SmallVector<float> hostIn0(length), hostIn1(length),
      hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length)) {
    hostIn0[i] = static_cast<float>(i);
    hostIn1[i] = static_cast<float>(length - i);
  }

  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  EXPECT_NO_ERR(bufIn0->copyFrom(hostIn0.data()));
  EXPECT_NO_ERR(bufIn1->copyFrom(hostIn1.data()));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  DeviceFunctionLaunchParams params;
  params.gridX = gridDim;
  params.gridY = 1;
  params.gridZ = 1;
  params.blockX = blockDim;
  params.blockY = 1;
  params.blockZ = 1;
  params.sharedMemBytes = 0;
  params.args = args;
  params.argCount = 4;

  EXPECT_NO_ERR(
      builder->addFunctionDirect(*deviceFunc, params, /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(bufOut->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn0[i] + hostIn1[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyHostToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(i);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyHostToDevice(*devBuf, hostIn.data(),
                                             /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToHost) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(2 * i + 1);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(devBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(hostOut.data(), *devBuf,
                                             /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(i * i);

  ASSIGN_NO_ERR(DeviceBufferRef, srcBuf, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, dstBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(srcBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(
      builder->addCopyDeviceToDevice(*dstBuf, *srcBuf, /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(dstBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

// Emitted kernel code only has the recording context in hand, so tensor
// pointers get wrapped against it (createBufferOwning stamps the wrapping
// context). The builder must resolve such a decorating context to its backing
// context instead of downcasting the decorator.
TEST(DeviceGraphBuilderTest, RecordedCopyDeviceToDeviceWithWrappedBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(3 * i + 2);

  ASSIGN_NO_ERR(DeviceBufferRef, srcBuf, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, dstBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(srcBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceGraphNode, root, builder->addEmpty(/*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceContextRef, recCtx, builder->recordingContext(root));

  DeviceBufferRef wrappedSrc =
      recCtx->createBufferOwning(srcBuf->ptr(), byteSize, /*owning=*/false);
  DeviceBufferRef wrappedDst =
      recCtx->createBufferOwning(dstBuf->ptr(), byteSize, /*owning=*/false);
  EXPECT_NO_ERR(recCtx->enqueueCopyDeviceToDevice(*wrappedDst, *wrappedSrc));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(dstBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddSetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));
  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);

  // Exercise valSize = 1, 2, 4, and symmetric 8 in a single graph by
  // recording the full sequence and replaying once. Each subsequent set
  // overwrites the previous one, so the four nodes are chained via explicit
  // dependencies and verification is against the final state.
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceGraphNode, n0,
                builder->addSetMemory(*devBuf, /*val=*/123, /*valSize=*/1,
                                      /*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphNode, n1,
                builder->addSetMemory(*devBuf, /*val=*/0xbeef, /*valSize=*/2,
                                      llvm::ArrayRef<DeviceGraphNode>(n0)));
  ASSIGN_NO_ERR(DeviceGraphNode, n2,
                builder->addSetMemory(*devBuf, /*val=*/0xdeadbeefu,
                                      /*valSize=*/4,
                                      llvm::ArrayRef<DeviceGraphNode>(n1)));
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/0x0101010101010101ull,
                                      /*valSize=*/8,
                                      llvm::ArrayRef<DeviceGraphNode>(n2)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostBuf.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // Final state is the symmetric 8-byte pattern (0x01 in every byte).
  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostBuf[i], 0x01u) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddSetMemoryRejectsAsymmetric8Byte) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 64;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  // High and low 32-bit halves differ; cannot be expressed as a single node.
  EXPECT_ERR_MSG(builder->addSetMemory(*devBuf,
                                       /*val=*/0x0123456789abcdefull,
                                       /*valSize=*/8, /*dependencies=*/{}),
                 "single graph node");
}

// Two graph operations that mutate the same buffer with different values.
// Without an explicit dependency edge from the memset to the copy, the two
// writes are graph roots and may execute concurrently — the final state
// would be a race. Adding the copy with the memset as its explicit
// predecessor forces the copy to run after the memset and the buffer must
// hold the copy's pattern.
TEST(DeviceGraphBuilderTest, ExplicitDepSequencesOpsOnSameBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 4096;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  llvm::SmallVector<uint8_t> hostIn(byteSize, 0xBBu);
  llvm::SmallVector<uint8_t> hostOut(byteSize, 0);

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceGraphNode, memsetNode,
                builder->addSetMemory(*devBuf, /*val=*/0xAAu, /*valSize=*/1,
                                      /*dependencies=*/{}));
  EXPECT_NO_ERR(builder->addCopyHostToDevice(
      *devBuf, hostIn.data(), llvm::ArrayRef<DeviceGraphNode>(memsetNode)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // The copy must observe a fully-written memset and overwrite every byte.
  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostOut[i], 0xBBu) << "at index " << i;
}

// Build two independent sequences of memsets, each on its own buffer. Each
// sequence's first node is rooted with an empty dependency list; subsequent
// nodes within a sequence carry an explicit single-element dependency on
// their immediate predecessor. The final state of each buffer must reflect
// the LAST memset of its own sequence — the two sequences are unordered
// with respect to each other but each is internally ordered.
TEST(DeviceGraphBuilderTest, IndependentMemsetSequences) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, bufA, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufB, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  // Empty ArrayRef = explicit "no dependencies" (graph root).
  llvm::ArrayRef<DeviceGraphNode> rootDeps;

  // Sequence A: 0x11 -> 0x22 -> 0x33 on bufA.
  ASSIGN_NO_ERR(
      DeviceGraphNode, a0,
      builder->addSetMemory(*bufA, /*val=*/0x11u, /*valSize=*/1, rootDeps));
  ASSIGN_NO_ERR(DeviceGraphNode, a1,
                builder->addSetMemory(*bufA, /*val=*/0x22u, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(a0)));
  EXPECT_NO_ERR(builder->addSetMemory(*bufA, /*val=*/0x33u, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(a1)));

  // Sequence B: 0xAA -> 0xBB -> 0xCC on bufB. Independent of sequence A.
  ASSIGN_NO_ERR(
      DeviceGraphNode, b0,
      builder->addSetMemory(*bufB, /*val=*/0xAAu, /*valSize=*/1, rootDeps));
  ASSIGN_NO_ERR(DeviceGraphNode, b1,
                builder->addSetMemory(*bufB, /*val=*/0xBBu, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(b0)));
  EXPECT_NO_ERR(builder->addSetMemory(*bufB, /*val=*/0xCCu, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(b1)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  llvm::SmallVector<uint8_t> hostA(byteSize, 0);
  llvm::SmallVector<uint8_t> hostB(byteSize, 0);
  EXPECT_NO_ERR(bufA->copyTo(hostA.data()));
  EXPECT_NO_ERR(bufB->copyTo(hostB.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostA[i], 0x33u) << "bufA at index " << i;

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostB[i], 0xCCu) << "bufB at index " << i;
}

// Empty nodes do no work but carry transitive ordering. Two independent
// memset roots fan in to a single empty node, and two host-to-device copies
// fan out from that empty node. Each copy is explicitly ordered after the
// empty node only — never against the memsets directly. If the empty node
// fails to propagate the dependency edges, the copies could race with the
// memsets and a final buffer would hold the memset pattern instead of the
// copy pattern.
TEST(DeviceGraphBuilderTest, EmptyNodeFanInFanOutJoin) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, bufA, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufB, ctx->enqueueCreateBuffer(byteSize));

  llvm::SmallVector<uint8_t> patternA(byteSize, 0xCCu);
  llvm::SmallVector<uint8_t> patternB(byteSize, 0xDDu);

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  // Fan-in: two independent memset roots.
  ASSIGN_NO_ERR(DeviceGraphNode, memsetA,
                builder->addSetMemory(*bufA, /*val=*/0xAAu, /*valSize=*/1,
                                      /*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphNode, memsetB,
                builder->addSetMemory(*bufB, /*val=*/0xBBu, /*valSize=*/1,
                                      /*dependencies=*/{}));

  // Join via an empty node.
  DeviceGraphNode fanInDeps[] = {memsetA, memsetB};
  ASSIGN_NO_ERR(DeviceGraphNode, joinNode, builder->addEmpty(fanInDeps));

  // Fan-out: two copies that depend only on the empty node. Transitive
  // ordering through the empty node must keep them after both memsets.
  EXPECT_NO_ERR(builder->addCopyHostToDevice(
      *bufA, patternA.data(), llvm::ArrayRef<DeviceGraphNode>(joinNode)));
  EXPECT_NO_ERR(builder->addCopyHostToDevice(
      *bufB, patternB.data(), llvm::ArrayRef<DeviceGraphNode>(joinNode)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  llvm::SmallVector<uint8_t> hostA(byteSize, 0);
  llvm::SmallVector<uint8_t> hostB(byteSize, 0);
  EXPECT_NO_ERR(bufA->copyTo(hostA.data()));
  EXPECT_NO_ERR(bufB->copyTo(hostB.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostA[i], 0xCCu) << "bufA at index " << i;
  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostB[i], 0xDDu) << "bufB at index " << i;
}

// A solo empty node with no dependencies and no successors must still
// instantiate and replay without error: an empty node is a valid graph root.
TEST(DeviceGraphBuilderTest, EmptyNodeAsSoleGraphRoot) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addEmpty(/*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());
}

// Memory from the builder's pool is ordinary device memory: graph nodes can
// write it and the recorded address stays valid across repeated replays.
TEST(DeviceGraphBuilderTest, CreateBufferIsWritableAcrossReplays) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceBufferRef, poolBuf,
                builder->createBuffer(byteSize, /*isHost=*/false));
  EXPECT_EQ(poolBuf->byteSize(), byteSize);

  EXPECT_NO_ERR(builder->addSetMemory(*poolBuf, /*val=*/0x5Au, /*valSize=*/1,
                                      /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);
  for (int replayCount : llvm::seq(2)) {
    EXPECT_NO_ERR(graph->replay());
    EXPECT_NO_ERR(ctx->synchronize());

    EXPECT_NO_ERR(poolBuf->copyTo(hostBuf.data()));
    EXPECT_NO_ERR(ctx->synchronize());

    for (size_t i : llvm::seq(byteSize))
      EXPECT_EQ(hostBuf[i], 0x5Au)
          << "at index " << i << " on replay " << replayCount;
  }
}

// The reason DeviceGraphMemoryPool exists: a graph bakes the raw device
// address of every allocation into its recorded nodes, so that address must
// stay reserved for the graph's lifetime even after the last `DeviceBuffer`
// handle is gone. Dropping the handle returns the block to the pool, which the
// graph keeps alive, so a later replay must still write valid memory. If the
// pool released the block to the parent free list (or to the driver), this
// replay would clobber memory owned by someone else.
TEST(DeviceGraphBuilderTest, PoolAllocationOutlivesBufferHandle) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceBufferRef, poolBuf,
                builder->createBuffer(byteSize, /*isHost=*/false));

  // The graph writes the pool buffer, then copies it back out, so verification
  // does not need the buffer handle to survive.
  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);
  ASSIGN_NO_ERR(DeviceGraphNode, memsetNode,
                builder->addSetMemory(*poolBuf, /*val=*/0xA5u, /*valSize=*/1,
                                      /*dependencies=*/{}));
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(
      hostBuf.data(), *poolBuf, llvm::ArrayRef<DeviceGraphNode>(memsetNode)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  // Drop the only handle to the allocation while the graph still references it.
  poolBuf = {};

  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostBuf[i], 0xA5u) << "at index " << i;
}

// A host request comes from the pool's other memory manager, so the memory it
// hands back must be writable by the host -- that is the whole reason a host
// pool exists. As with the device case, the address must stay reserved once the
// only `DeviceBuffer` handle is gone: the graph records it as a copy source, so
// a replay after the handle dies must still read what the host wrote.
TEST(DeviceGraphBuilderTest, HostPoolAllocationOutlivesBufferHandle) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf,
                builder->createBuffer(byteSize, /*isHost=*/true));
  EXPECT_EQ(hostBuf->byteSize(), byteSize);
  EXPECT_TRUE(hostBuf->isHost());

  std::memset(hostBuf->ptr(), 0x3Cu, byteSize);

  // Stage the host allocation into device memory, then back out to storage the
  // test owns, so verification does not depend on either pool buffer surviving.
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf,
                builder->createBuffer(byteSize, /*isHost=*/false));
  llvm::SmallVector<uint8_t> result(byteSize, 0);
  ASSIGN_NO_ERR(DeviceGraphNode, copyIn,
                builder->addCopyHostToDevice(*devBuf, hostBuf->ptr(),
                                             /*dependencies=*/{}));
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(
      result.data(), *devBuf, llvm::ArrayRef<DeviceGraphNode>(copyIn)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  // Drop the handles while the graph still references both addresses.
  hostBuf = {};
  devBuf = {};

  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(result[i], 0x3Cu) << "at index " << i;
}

// A zero-byte request is a placeholder, not an allocation: the pool's allocator
// rejects a zero size, so `createBuffer` must short-circuit the way
// `enqueueCreateBuffer` does rather than reaching it. Neither pool is exempt.
TEST(DeviceGraphBuilderTest, CreateBufferAcceptsZeroBytes) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf,
                builder->createBuffer(0, /*isHost=*/false));
  EXPECT_EQ(emptyBuf->byteSize(), 0u);
  ASSIGN_NO_ERR(DeviceBufferRef, emptyHostBuf,
                builder->createBuffer(0, /*isHost=*/true));
  EXPECT_EQ(emptyHostBuf->byteSize(), 0u);

  // Instantiate and replay to confirm the placeholder buffer does not disturb
  // pool teardown when the graph and buffer are released.
  EXPECT_NO_ERR(builder->addEmpty(/*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());
}

// The pool is handed to the graph at `instantiate()`, so the builder can no
// longer serve allocations: `createBuffer` must report that rather than
// dereferencing the moved-from pool.
TEST(DeviceGraphBuilderTest, CreateBufferAfterInstantiateFails) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addEmpty(/*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  EXPECT_TRUE(builder->createBuffer(1024, /*isHost=*/false).isError());
}

// In-place input markers (mutable inputs the graph writes through the
// caller's live address) occupy a position in the inputs list as null entries,
// interleaved with stable-buffer entries in registration order. The replay
// copy in `mgp_device_graph_execute` relies on that positional pairing to
// skip exactly the in-place positions.
TEST(DeviceGraphBuilderTest, AddInPlaceInputInterleavesWithStableInputs) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  ASSIGN_NO_ERR(DeviceBufferRef, stable0,
                builder->createBuffer(1024, /*isHost=*/false));
  builder->addInput(stable0.copy());
  EXPECT_EQ(builder->numInputs(), 1u);

  builder->addInPlaceInput();
  EXPECT_EQ(builder->numInputs(), 2u);

  ASSIGN_NO_ERR(DeviceBufferRef, stable2,
                builder->createBuffer(512, /*isHost=*/false));
  builder->addInput(stable2.copy());
  EXPECT_EQ(builder->numInputs(), 3u);

  EXPECT_NO_ERR(builder->addEmpty(/*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  llvm::ArrayRef<DeviceBufferRef> inputs = graph->getInputs();
  ASSERT_EQ(inputs.size(), 3u);
  EXPECT_TRUE(inputs[0]);
  EXPECT_EQ(inputs[0]->ptr(), stable0->ptr());
  EXPECT_FALSE(inputs[1]);
  EXPECT_TRUE(inputs[2]);
  EXPECT_EQ(inputs[2]->ptr(), stable2->ptr());
}

// Slot assignments recorded on the builder move to the graph at
// `instantiate()`, and `patchStableAddresses` writes each value into its slot
// on the graph's context. The write is asynchronous, so the read-back waits on
// the context first, as `mgp.device_graph.collective.execute` does before
// launching any graph.
TEST(DeviceGraphBuilderTest, StableAssignmentsPatchIntoSlots) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceBufferRef, slot,
                ctx->enqueueCreateBuffer(sizeof(void *)));
  EXPECT_NO_ERR(ctx->enqueueSetMemory(*slot, /*val=*/0, /*valSize=*/1));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceBufferRef, value,
                builder->createBuffer(1024, /*isHost=*/false));
  builder->addStableAssignment(static_cast<void **>(slot->ptr()), value->ptr());
  EXPECT_EQ(builder->numStableAssignments(), 1u);

  EXPECT_NO_ERR(builder->addEmpty(/*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_EQ(builder->numStableAssignments(), 0u);
  EXPECT_EQ(&graph->context(), ctx.getPointer());

  llvm::ArrayRef<DeviceGraph::StableAssignment> assignments =
      graph->getStableAssignments();
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments[0].stableAddress, slot->ptr());
  EXPECT_EQ(assignments[0].ptr, value->ptr());

  EXPECT_NO_ERR(graph->patchStableAddresses());
  EXPECT_NO_ERR(ctx->synchronize());

  void *hostSlot = nullptr;
  EXPECT_NO_ERR(slot->copyTo(&hostSlot));
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_EQ(hostSlot, value->ptr());
}

// A builder can allocate from an externally owned pool instead of a private
// one. The graph still takes the builder's pool reference at `instantiate()`,
// but the external owner's reference keeps the pool usable for further
// builders -- that is what lets multiple graphs share one pool.
TEST(DeviceGraphBuilderTest, BuilderWithExternalPoolReplays) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  DeviceGraphMemoryPoolRef pool = DeviceGraphMemoryPoolRef::create(ctx.copy());

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder,
                ctx->createGraphBuilder(pool.copy()));
  ASSIGN_NO_ERR(DeviceBufferRef, poolBuf,
                builder->createBuffer(byteSize, /*isHost=*/false));

  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);
  ASSIGN_NO_ERR(DeviceGraphNode, memsetNode,
                builder->addSetMemory(*poolBuf, /*val=*/0xA5u, /*valSize=*/1,
                                      /*dependencies=*/{}));
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(
      hostBuf.data(), *poolBuf, llvm::ArrayRef<DeviceGraphNode>(memsetNode)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());

  // Drop the only handle to the allocation while the graph still references
  // it, as PoolAllocationOutlivesBufferHandle does for a private pool.
  poolBuf = {};

  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostBuf[i], 0xA5u) << "at index " << i;

  // `instantiate()` consumed only the builder's reference: the external one
  // still serves a second builder.
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder2,
                ctx->createGraphBuilder(pool.copy()));
  ASSIGN_NO_ERR(DeviceBufferRef, poolBuf2,
                builder2->createBuffer(byteSize, /*isHost=*/false));
  EXPECT_EQ(poolBuf2->byteSize(), byteSize);
}

// The point of sharing a pool: a block freed by one builder is reissued to the
// next, so their graphs overlay the same transient memory. Graph A's replay
// then writes memory graph B also uses, which is sound exactly because graphs
// sharing a pool replay serially in recording order.
TEST(DeviceGraphBuilderTest, SharedPoolReissuesFreedTransientAcrossBuilders) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  DeviceGraphMemoryPoolRef pool = DeviceGraphMemoryPoolRef::create(ctx.copy());

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builderA,
                ctx->createGraphBuilder(pool.copy()));
  ASSIGN_NO_ERR(DeviceBufferRef, bufA,
                builderA->createBuffer(byteSize, /*isHost=*/false));
  void *sharedAddr = bufA->ptr();
  EXPECT_NO_ERR(builderA->addSetMemory(*bufA, /*val=*/0x11u, /*valSize=*/1,
                                       /*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphRef, graphA, builderA->instantiate());

  // Return the block to the shared pool; the graph keeps it reserved, but the
  // free list may now reissue the address.
  bufA = {};

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builderB,
                ctx->createGraphBuilder(pool.copy()));
  ASSIGN_NO_ERR(DeviceBufferRef, bufB,
                builderB->createBuffer(byteSize, /*isHost=*/false));
  EXPECT_EQ(bufB->ptr(), sharedAddr)
      << "same-size allocation from the shared pool should reuse the freed "
         "block";

  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);
  ASSIGN_NO_ERR(DeviceGraphNode, memsetB,
                builderB->addSetMemory(*bufB, /*val=*/0x22u, /*valSize=*/1,
                                       /*dependencies=*/{}));
  EXPECT_NO_ERR(builderB->addCopyDeviceToHost(
      hostBuf.data(), *bufB, llvm::ArrayRef<DeviceGraphNode>(memsetB)));
  ASSIGN_NO_ERR(DeviceGraphRef, graphB, builderB->instantiate());

  // Serial replay in recording order: A writes the shared block, then B
  // overwrites it and copies its own value out.
  EXPECT_NO_ERR(graphA->replay());
  EXPECT_NO_ERR(graphB->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostBuf[i], 0x22u) << "at index " << i;
}

} // namespace
