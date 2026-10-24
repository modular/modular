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

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "mlir/Pass/Pass.h"

namespace M::MAP {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

class IterElementwiseOp;

/// Whether `RaiseBackContiguousViews` turns `op` into zero-copy views, so it
/// never runs as a kernel.
bool isRaisableViewChain(IterElementwiseOp op);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H
