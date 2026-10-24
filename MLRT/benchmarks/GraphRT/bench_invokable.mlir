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
// Benchmark MLIR for measuring invokable call overhead with async operations.
// Derived from MLRT/test/GraphRT/mef-executor/fibonacci.mlir
//
//===----------------------------------------------------------------------===//

// Base case - minimal invokable (returns chain for synchronization)
mef.callable @noop(%ch: !mef.chain) -> !mef.chain {
  mef.output %ch : !mef.chain
}

// Call depth 1
mef.callable @call_depth_1(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @noop(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 2
mef.callable @call_depth_2(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_1(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 3
mef.callable @call_depth_3(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_2(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 4
mef.callable @call_depth_4(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_3(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 5
mef.callable @call_depth_5(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_4(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 6
mef.callable @call_depth_6(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_5(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 7
mef.callable @call_depth_7(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_6(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 8
mef.callable @call_depth_8(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_7(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 9
mef.callable @call_depth_9(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_8(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

// Call depth 10
mef.callable @call_depth_10(%ch: !mef.chain) -> !mef.chain {
  %r = grt.invoke @call_depth_9(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

//===----------------------------------------------------------------------===//
// Testcases for different call depths
//===----------------------------------------------------------------------===//

grt.testcase @bench_call_depth_1 {
  %ch = grt.chain.create ()
  %r = grt.invoke @call_depth_1(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}


grt.testcase @bench_call_depth_10 {
  %ch = grt.chain.create ()
  %r = grt.invoke @call_depth_10(%ch) : (!mef.chain) -> !mef.chain
  mef.output %r : !mef.chain
}

//===----------------------------------------------------------------------===//
// Invokable with arguments and return values
//===----------------------------------------------------------------------===//

mef.callable @identity(%x: si32) -> si32 {
  mef.output %x : si32
}

mef.callable @add_one(%x: si32) -> si32 {
  %one = grt.constant.si32 1
  %r = "test.async.add.i32"(%x, %one) : (si32, si32) -> si32
  mef.output %r : si32
}

// Chain of invokable calls with values
mef.callable @chain_call_5(%x: si32) -> si32 {
  %r1 = grt.invoke @add_one(%x) : (si32) -> si32
  %r2 = grt.invoke @add_one(%r1) : (si32) -> si32
  %r3 = grt.invoke @add_one(%r2) : (si32) -> si32
  %r4 = grt.invoke @add_one(%r3) : (si32) -> si32
  %r5 = grt.invoke @add_one(%r4) : (si32) -> si32
  mef.output %r5 : si32
}

grt.testcase @bench_chain_call_5 {
  %x = grt.constant.si32 0
  %r = grt.invoke @chain_call_5(%x) : (si32) -> si32
  mef.output %r : si32
}
