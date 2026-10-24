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

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindMODialect(nb::module_ m) {
  using namespace M::MO;
/// Bind types first, they're used in defaults for op builders
#include "GraphCompiler/MODialect/Nanobind/MOTypes.cpp.inc"

#include "GraphCompiler/MODialect/Nanobind/MOAttrs.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOCompositeOps.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOEnums.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOInterfaces.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOOps.cpp.inc"

  nb::class_<ParamExprBuilder>(m, "ParamExprBuilder");
  nb::class_<ShapeMaterializeResult>(m, "ShapeMaterializeResult");
  nb::module_ passes = m.def_submodule("passes");
  {
    nb::module_ m = passes;
#include "GraphCompiler/MODialect/Nanobind/MOTransforms.cpp.inc"
  }
}
