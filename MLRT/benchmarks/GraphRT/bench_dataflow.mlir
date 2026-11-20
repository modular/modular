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
// Benchmark MLIR for measuring async dataflow scheduling overhead.
// Derived from MLRT/test/GraphRT/mef-executor/async.mlir
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parallel Independent Operations
// These test the scheduler's ability to execute independent ops concurrently
//===----------------------------------------------------------------------===//

// 2 independent operations that can run in parallel
grt.testcase @bench_parallel_2 {
  %c1 = grt.constant.si32 1
  %c2 = grt.constant.si32 2
  // Independent adds
  %a = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %b = "test.async.add.i32"(%c2, %c2) : (si32, si32) -> si32
  // Join
  %r = "test.async.add.i32"(%a, %b) : (si32, si32) -> si32
  mef.output %r : si32
}

// 8 independent operations that can run in parallel
grt.testcase @bench_parallel_8 {
  %c1 = grt.constant.si32 1
  // 8 independent adds
  %a = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %b = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %c = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %d = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %e = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %f = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %g = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %h = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  // Join tree - level 1
  %ab = "test.async.add.i32"(%a, %b) : (si32, si32) -> si32
  %cd = "test.async.add.i32"(%c, %d) : (si32, si32) -> si32
  %ef = "test.async.add.i32"(%e, %f) : (si32, si32) -> si32
  %gh = "test.async.add.i32"(%g, %h) : (si32, si32) -> si32
  // Join tree - level 2
  %abcd = "test.async.add.i32"(%ab, %cd) : (si32, si32) -> si32
  %efgh = "test.async.add.i32"(%ef, %gh) : (si32, si32) -> si32
  // Final join
  %r = "test.async.add.i32"(%abcd, %efgh) : (si32, si32) -> si32
  mef.output %r : si32
}

//===----------------------------------------------------------------------===//
// Sequential Dependencies
// Baseline for comparison with parallel execution
//===----------------------------------------------------------------------===//

// Strictly sequential chain of operations
grt.testcase @bench_sequential_8 {
  %c1 = grt.constant.si32 1
  %r1 = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %r2 = "test.async.add.i32"(%r1, %c1) : (si32, si32) -> si32
  %r3 = "test.async.add.i32"(%r2, %c1) : (si32, si32) -> si32
  %r4 = "test.async.add.i32"(%r3, %c1) : (si32, si32) -> si32
  %r5 = "test.async.add.i32"(%r4, %c1) : (si32, si32) -> si32
  %r6 = "test.async.add.i32"(%r5, %c1) : (si32, si32) -> si32
  %r7 = "test.async.add.i32"(%r6, %c1) : (si32, si32) -> si32
  %r8 = "test.async.add.i32"(%r7, %c1) : (si32, si32) -> si32
  mef.output %r8 : si32
}

//===----------------------------------------------------------------------===//
// Multiple Independent Invokable Calls
// Tests parallel scheduling of invokable invocations
//===----------------------------------------------------------------------===//

mef.callable @add_self(%x: si32) -> si32 {
  %r = "test.async.add.i32"(%x, %x) : (si32, si32) -> si32
  mef.output %r : si32
}

// 8 independent invokable calls that can execute in parallel
grt.testcase @bench_parallel_invoke_8 {
  %c1 = grt.constant.si32 1
  %c2 = grt.constant.si32 2
  %c3 = grt.constant.si32 3
  %c4 = grt.constant.si32 4
  %c5 = grt.constant.si32 5
  %c6 = grt.constant.si32 6
  %c7 = grt.constant.si32 7
  %c8 = grt.constant.si32 8
  // Independent calls
  %r1 = grt.invoke @add_self(%c1) : (si32) -> si32
  %r2 = grt.invoke @add_self(%c2) : (si32) -> si32
  %r3 = grt.invoke @add_self(%c3) : (si32) -> si32
  %r4 = grt.invoke @add_self(%c4) : (si32) -> si32
  %r5 = grt.invoke @add_self(%c5) : (si32) -> si32
  %r6 = grt.invoke @add_self(%c6) : (si32) -> si32
  %r7 = grt.invoke @add_self(%c7) : (si32) -> si32
  %r8 = grt.invoke @add_self(%c8) : (si32) -> si32
  // Join
  %j1 = "test.async.add.i32"(%r1, %r2) : (si32, si32) -> si32
  %j2 = "test.async.add.i32"(%r3, %r4) : (si32, si32) -> si32
  %j3 = "test.async.add.i32"(%r5, %r6) : (si32, si32) -> si32
  %j4 = "test.async.add.i32"(%r7, %r8) : (si32, si32) -> si32
  %j12 = "test.async.add.i32"(%j1, %j2) : (si32, si32) -> si32
  %j34 = "test.async.add.i32"(%j3, %j4) : (si32, si32) -> si32
  %r = "test.async.add.i32"(%j12, %j34) : (si32, si32) -> si32
  mef.output %r : si32
}

//===----------------------------------------------------------------------===//
// GPU Simultaneous Ready Primitives
// Tests MEF executor serialization overhead when multiple primitives become
// ready at the same time due to sharing the same inputs.
//
// This simulates the multi-device allreduce scheduling problem where N
// allreduce ops on different devices share the same input tensors, causing
// them to become ready simultaneously and be launched serially (~3us each).
//
// Uses test.sleep to simulate ~3us GPU kernel launch latency.
//===----------------------------------------------------------------------===//

// 2 GPU kernel launches become ready simultaneously (share same input chain)
grt.testcase @bench_gpu_simultaneous_ready_2 {
  %ch0 = grt.chain.create ()
  // Both ops share the same input chain - they become ready at the same time
  // Each simulates a ~3us GPU kernel launch (3000ns)
  %ch1 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch2 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  // Join all chains
  %ch_final = grt.chain.create (%ch1, %ch2)
  mef.output %ch_final : !mef.chain
}

// 8 GPU kernel launches become ready simultaneously (simulates 8-GPU tensor parallel)
grt.testcase @bench_gpu_simultaneous_ready_8 {
  %ch0 = grt.chain.create ()
  // All 8 ops share the same input chain - simultaneous ready
  %ch1 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch2 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch3 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch4 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch5 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch6 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch7 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  %ch8 = "test.sleep"(%ch0) { delay_ns = 3000 : si32 } : (!mef.chain) -> !mef.chain
  // Join all chains
  %ch_final = grt.chain.create (%ch1, %ch2, %ch3, %ch4, %ch5, %ch6, %ch7, %ch8)
  mef.output %ch_final : !mef.chain
}
