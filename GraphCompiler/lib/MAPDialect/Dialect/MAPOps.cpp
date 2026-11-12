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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// The declarative formats name these unqualified.
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

using namespace M;
using namespace mlir;
using namespace M::MAP;
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

static bool isStaticOne(Attribute dim) {
  return MOSH::ShapeAttr::getStaticDim(dim) == 1;
}

// numpy broadcast: `in`'s missing leading dims are dropped, its size-1 dims
// pinned.
std::optional<SmallVector<int64_t>>
BroadcastIndexOp::mapLoopDims(ArrayRef<int64_t> dims) {
  ViewSignatureAttr signature = getSpec().getSignature();
  MOSH::ShapeAttr in = signature.getIn(), out = signature.getOut();
  int64_t lead = out.getRank() - in.getRank();
  SmallVector<int64_t> result;
  for (int64_t d = 0, e = in.getRank(); d < e; ++d) {
    if (in[d] == out[d + lead])
      result.push_back(dims[d + lead]);
    else if (isStaticOne(in[d]))
      result.push_back(kPinnedCoordinate);
    else
      return std::nullopt;
  }
  return result;
}

// Only a reshape that adds or drops unit dims maps coordinates one to one.
std::optional<SmallVector<int64_t>>
ReshapeIndexOp::mapLoopDims(ArrayRef<int64_t> dims) {
  ViewSignatureAttr signature = getSpec().getSignature();
  MOSH::ShapeAttr in = signature.getIn(), out = signature.getOut();
  auto nonUnitDims = [](MOSH::ShapeAttr shape) {
    SmallVector<int64_t> dims;
    for (int64_t d = 0, e = shape.getRank(); d < e; ++d)
      if (!isStaticOne(shape[d]))
        dims.push_back(d);
    return dims;
  };
  SmallVector<int64_t> inDims = nonUnitDims(in), outDims = nonUnitDims(out);
  if (inDims.size() != outDims.size())
    return std::nullopt;
  SmallVector<int64_t> result(in.getRank(), kPinnedCoordinate);
  for (auto [inDim, outDim] : llvm::zip(inDims, outDims)) {
    if (in[inDim] != out[outDim])
      return std::nullopt;
    result[inDim] = dims[outDim];
  }
  return result;
}

std::optional<SmallVector<int64_t>>
TransposeIndexOp::mapLoopDims(ArrayRef<int64_t> dims) {
  ArrayRef<int64_t> perm = getSpec().getPerm();
  SmallVector<int64_t> result(perm.size());
  for (auto [i, p] : llvm::enumerate(perm))
    result[p] = dims[i];
  return result;
}

//===----------------------------------------------------------------------===//
// map.store queries
//===----------------------------------------------------------------------===//

StoreOp M::MAP::getUniqueStore(Block &body) {
  SmallVector<StoreOp> stores = getOutputStores(body);
  ASSERT_STREAM(stores.size() == 1,
                << "expected exactly one map.store in the body");
  // A store nested in a region runs only for some elements.
  ASSERT_STREAM(stores.front()->getBlock() == &body,
                << "expected the map.store directly in the body");
  return stores.front();
}

SmallVector<StoreOp> M::MAP::getOutputStores(Block &body) {
  SmallVector<StoreOp> stores;
  for (Operation &op : body)
    op.walk<WalkOrder::PreOrder>(
        [&](StoreOp store) { stores.push_back(store); });
  ASSERT_STREAM(!stores.empty(), << "expected a map.store in the body");
  return stores;
}

#define GET_OP_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
