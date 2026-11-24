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

#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace NB_NAMESPACE {
namespace detail {

/// Casts object <-> M::KGEN::FnEffects.
/// - C++ has M::KGEN::impl::FnEffects and M::KGEN::FnEffects
/// - We just use one for simplicity, and standardize on the generated one.
template <>
struct type_caster<M::KGEN::FnEffects> {
  using Caster = make_caster<M::KGEN::impl::FnEffects>;
  NB_TYPE_CASTER(M::KGEN::FnEffects, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = caster.value;
      return true;
    }
    return false;
  }
  static handle from_cpp(M::KGEN::FnEffects t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(t.getImpl(), policy, cleanup);
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

void M::Graph::Python::bindKGENDialect(nb::module_ m) {
  using namespace mlir;
  using namespace M::KGEN;

// Enums are included first so that their `nb::enum_<>` registrations land
// before any attribute binding that uses an enum value as a default argument
// (e.g. `PogMetadataAttr` defaults the `variadic` arg to `VariadicKind::None`,
// which requires the Python type for `VariadicKind` to already exist when
// nanobind casts the default).
#include "Mojo/KGENDialect/Nanobind/KGENEnums.cpp.inc"

#include "Mojo/KGENDialect/Nanobind/KGENAttrInterfaces.cpp.inc"
#include "Mojo/KGENDialect/Nanobind/KGENAttrs.cpp.inc"
#include "Mojo/KGENDialect/Nanobind/KGENOps.cpp.inc"
#include "Mojo/KGENDialect/Nanobind/KGENTypeInterfaces.cpp.inc"
#include "Mojo/KGENDialect/Nanobind/KGENTypes.cpp.inc"

  nb::class_<FuncTypeGeneratorType, GeneratorType>(m, "FuncTypeGeneratorType")
      .def(nb::new_(nb::overload_cast<ArrayRef<Type>, FunctionType,
                                      ArrayRef<ArgConvention>, FnEffects,
                                      Attribute, Attribute, Attribute>(
               &FuncTypeGeneratorType::get)),
           nb::arg("input_param_types"), nb::arg("fn_type"),
           nb::arg("arg_convs") = ArrayRef<ArgConvention>(),
           nb::arg("effects") = FnEffects(),
           nb::arg("fn_metadata") = Attribute(),
           nb::arg("gen_metadata") = Attribute(),
           nb::arg("arg_list_attrs") = Attribute());

  nb::class_<KGENDType>(m, "_KGENDType")
      .def_static("get_int",
                  [](int64_t widthInBits, bool isSigned) -> KGENDType {
                    auto result = KGENDType::getInt(widthInBits, isSigned);
                    assert(succeeded(result) && "Failed to get dtype");
                    return KGENDType(*result);
                  });

  nb::class_<DTypeValue>(m, "_DTypeValue");
  nb::class_<ParamDefValue>(m, "ParamDefValue");
  nb::class_<ParameterEvaluator>(m, "ParameterEvaluator");
}
