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

#include "GraphCompiler/MODialect/MOOps.h"
#include "Support/MDialect/MAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace M::MO {

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_matmul_b_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackMatmulShapeFunc(InvokeShapeFuncOp invoke,
                               TargetInfoAttr targetAttr);

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_filter_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackConvFilterShape(InvokeShapeFuncOp invoke,
                               TargetInfoAttr targetAttr);

std::optional<SmallVector<int64_t>>
materializePackConvTransposeFilterShape(InvokeShapeFuncOp invoke,
                                        TargetInfoAttr targetAttr);

} // namespace M::MO
