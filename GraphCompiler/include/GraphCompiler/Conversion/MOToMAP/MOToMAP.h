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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H

#include "GraphCompiler/MODialect/MOUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MOGG {
class MojoRegistry;
} // namespace M::MOGG

namespace M::MO {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h.inc"

/// Lowers a single MO kernel op to MAP in place, as `mo-to-map` does. Returns
/// the created `map.iter.opaque` on the opaque kernel path, null on the other
/// successful paths and for an op with no kernel adaptor (left unchanged), or
/// `failure()` on a lowering error.
///
/// Exposed for `NanCheckPass`, which injects `mo.custom` check kernels after
/// `mo-to-map` has run.
FailureOr<Operation *> lowerMOOpToMAP(Operation *op,
                                      MOGG::MojoRegistry &registry);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
