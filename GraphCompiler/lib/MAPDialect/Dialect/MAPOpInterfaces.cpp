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

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.cpp.inc"

#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.h"

std::optional<llvm::SmallVector<int64_t>>
M::MAP::IndexTransformOpInterface::traceLoopDims(
    mlir::Value index, mlir::Value loopIdx, std::optional<int64_t> rowAxis) {
  if (index == loopIdx)
    return llvm::to_vector(llvm::seq<int64_t>(
        0, mlir::cast<IndexType>(index.getType()).getRank()));

  // A concat select's region reads at its index, moved along the select's
  // axis.
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(index); arg && rowAxis) {
    auto select = mlir::dyn_cast<ConcatSelectOp>(arg.getOwner()->getParentOp());
    if (!select)
      return std::nullopt;
    std::optional<llvm::SmallVector<int64_t>> dims =
        traceLoopDims(select.getIndex(), loopIdx, rowAxis);
    if (!dims || (*dims)[select.getSpec().getAxis()] != *rowAxis)
      return std::nullopt;
    return dims;
  }

  auto transform = index.getDefiningOp<IndexTransformOpInterface>();
  if (!transform)
    return std::nullopt;
  std::optional<llvm::SmallVector<int64_t>> outer =
      traceLoopDims(transform.getIndex(), loopIdx, rowAxis);
  if (!outer)
    return std::nullopt;
  return transform.mapLoopDims(*outer, rowAxis);
}
