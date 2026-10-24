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

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace M::MO;

namespace {

static CallOp findCallOpByCallee(ModuleOp module, StringRef calleeName) {
  CallOp found;
  module.walk([&](CallOp call) {
    if (call.getCallee() == calleeName) {
      found = call;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Tests that mo.call ops with a single chain work with getIn/OutChain.
TEST(CallOpInterfaceTest, SingleChainCallProvidesSingleton) {
  constexpr StringLiteral kIR = R"mlir(
module {
  mo.graph @store_buffer(%buf: !mo.buffer<[1], f32>, %tensor: !mo.tensor<[1], f32>, %chain: !mo.chain)
      -> !mo.chain isSubgraph {
    %loaded, %ch1 = mo.mutable.load(%buf, %chain)
        : (!mo.buffer<[1], f32>, !mo.chain) -> (!mo.tensor<[1], f32>, !mo.chain)
    %sum = mo.add(%loaded, %tensor) : !mo.tensor<[1], f32>
    %ch2 = mo.mutable.store(%buf, %sum, %ch1)
        : (!mo.buffer<[1], f32>, !mo.tensor<[1], f32>, !mo.chain) -> (!mo.chain)
    mo.output %ch2 : !mo.chain
  }

  mo.graph @caller(%buf: !mo.buffer<[1], f32>, %tensor: !mo.tensor<[1], f32>)
      -> (!mo.buffer<[1], f32>) {
    %ch0 = mo.chain.create()
    %ch1 = mo.call @store_buffer(%buf, %tensor, %ch0)
        : (!mo.buffer<[1], f32>, !mo.tensor<[1], f32>, !mo.chain) -> !mo.chain
    mo.output %buf : !mo.buffer<[1], f32>
  }
}
)mlir";

  MLIRContext ctx;
  ctx.loadDialect<M::MO::MODialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kIR, &ctx);
  ASSERT_TRUE(module);

  CallOp call = findCallOpByCallee(*module, "store_buffer");
  ASSERT_TRUE(call);

  auto multi = dyn_cast<MOMutableOpInterface>(call.getOperation());
  ASSERT_TRUE(multi);

  SmallVector<ChainValue> inChains = multi.getInChains();
  ASSERT_EQ(inChains.size(), 1u);

  SmallVector<ChainValue> outChains = multi.getOutChains();
  ASSERT_EQ(outChains.size(), 1u);

  (void)call.getInChains();
  (void)call.getOutChains();
}

// Tests that MultiChainMutableOpInterface works with multi-chain mo.call ops.
TEST(CallOpInterfaceTest, MultipleChainsExposedThroughMultiInterface) {
  constexpr StringLiteral kIR = R"mlir(
module {
  mo.graph @callee_multi(%buf: !mo.buffer<[1], f32>, %chain_main: !mo.chain, %chain_dev: !mo.chain)
      -> (!mo.chain, !mo.chain) isSubgraph {
    mo.output %chain_main, %chain_dev : !mo.chain, !mo.chain
  }

  mo.graph @caller_multi(%buf: !mo.buffer<[1], f32>) -> (!mo.buffer<[1], f32>) {
    %ch0 = mo.chain.create()
    %ch1 = mo.chain.create()
    %main_out, %dev_out = mo.call @callee_multi(%buf, %ch0, %ch1)
        : (!mo.buffer<[1], f32>, !mo.chain, !mo.chain) -> (!mo.chain, !mo.chain)
    mo.output %buf : !mo.buffer<[1], f32>
  }
}
)mlir";

  MLIRContext ctx;
  ctx.loadDialect<M::MO::MODialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kIR, &ctx);
  ASSERT_TRUE(module);

  CallOp call = findCallOpByCallee(*module, "callee_multi");
  ASSERT_TRUE(call);

  auto multi = dyn_cast<MOMutableOpInterface>(call.getOperation());
  ASSERT_TRUE(multi);

  SmallVector<ChainValue> inChains = multi.getInChains();
  EXPECT_EQ(inChains.size(), 2u);

  SmallVector<ChainValue> outChains = multi.getOutChains();
  EXPECT_EQ(outChains.size(), 2u);
}

} // namespace
