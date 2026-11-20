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

#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindRMODialect(nb::module_ m) {
  using namespace M::RMO;
#include "GraphCompiler/RMODialect/Nanobind/RMOAttrs.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOInterfaces.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOMOAnalogueOps.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOOps.cpp.inc"

  nb::module_ passes = m.def_submodule("passes");
  {
    nb::module_ m = passes;
#include "GraphCompiler/RMODialect/Nanobind/RMOTransforms.cpp.inc"
  }
}
