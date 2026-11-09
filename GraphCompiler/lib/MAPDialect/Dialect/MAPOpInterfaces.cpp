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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.cpp.inc"

#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.h"

std::optional<llvm::SmallVector<int64_t>>
M::MAP::traceLoopDims(mlir::Value index, mlir::Value loopIdx) {
  if (index == loopIdx)
    return llvm::to_vector(llvm::seq<int64_t>(
        0, mlir::cast<IndexType>(index.getType()).getRank()));
  auto transform = index.getDefiningOp<IndexTransformOpInterface>();
  if (!transform)
    return std::nullopt;
  std::optional<llvm::SmallVector<int64_t>> outer =
      traceLoopDims(transform.getIndex(), loopIdx);
  return outer ? transform.mapLoopDims(*outer) : std::nullopt;
}
