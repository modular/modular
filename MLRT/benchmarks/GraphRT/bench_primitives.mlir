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
// Benchmark MLIR for measuring async primitive dispatch overhead.
// Derived from MLRT/test/GraphRT/mef-executor/basics.mlir
//
//===----------------------------------------------------------------------===//

// Baseline: schedules one async add but doesn't return the result.
// Used to measure pure scheduling overhead without waiting for completion.
grt.testcase @bench_primitive_baseline {
  %x = grt.constant.si32 1
  %r = "test.async.add.i32"(%x, %x) : (si32, si32) -> si32
  mef.output
}

// Single async integer add - measures single primitive dispatch cost
mef.callable @primitive_chain_1() -> si32 {
  %x = grt.constant.si32 1
  %r1 = "test.async.add.i32"(%x, %x) : (si32, si32) -> si32
  mef.output %r1 : si32
}

// Chain of 10 sequential async integer adds - measures primitive dispatch cost
mef.callable @primitive_chain_10() -> si32 {
  %x = grt.constant.si32 1
  %r1 = "test.async.add.i32"(%x, %x) : (si32, si32) -> si32
  %r2 = "test.async.add.i32"(%r1, %x) : (si32, si32) -> si32
  %r3 = "test.async.add.i32"(%r2, %x) : (si32, si32) -> si32
  %r4 = "test.async.add.i32"(%r3, %x) : (si32, si32) -> si32
  %r5 = "test.async.add.i32"(%r4, %x) : (si32, si32) -> si32
  %r6 = "test.async.add.i32"(%r5, %x) : (si32, si32) -> si32
  %r7 = "test.async.add.i32"(%r6, %x) : (si32, si32) -> si32
  %r8 = "test.async.add.i32"(%r7, %x) : (si32, si32) -> si32
  %r9 = "test.async.add.i32"(%r8, %x) : (si32, si32) -> si32
  %r10 = "test.async.add.i32"(%r9, %x) : (si32, si32) -> si32
  mef.output %r10 : si32
}

// Testcase: 1 primitive
grt.testcase @bench_primitive_1 {
  %r = grt.invoke @primitive_chain_1() : () -> si32
  mef.output %r : si32
}

// Testcase: 10 primitives
grt.testcase @bench_primitive_10 {
  %r = grt.invoke @primitive_chain_10() : () -> si32
  mef.output %r : si32
}
