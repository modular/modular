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

#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindMDialect(nb::module_ m) {
  using namespace M;
  using InOutSemantics = M::InOutSignatureAttr::InOutSemantics;

  nb::enum_<InOutSemantics>(m, "InOutSemantics")
      .value("none", InOutSemantics::kNone)
      .value("in_", InOutSemantics::kIn) // in is a python keyword
      .value("out", InOutSemantics::kOut)
      .value("mut", InOutSemantics::kMut);

#include "Support/MDialect/Nanobind/MAttrInterfaces.cpp.inc"
#include "Support/MDialect/Nanobind/MAttrs.cpp.inc"
#include "Support/MDialect/Nanobind/MTypes.cpp.inc"

  nb::class_<DataLayout>(m, "DataLayout");
  nb::class_<IntArrayElementsAttr>(m, "IntArrayElementsAttr");
}
