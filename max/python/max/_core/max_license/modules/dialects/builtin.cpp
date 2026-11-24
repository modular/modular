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

#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/Passes.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/APSInt.h"

namespace nb = nanobind;

void M::Graph::Python::bindBuiltinDialect(nb::module_ m) {
  using namespace mlir;
  using SignednessSemantics = IntegerType::SignednessSemantics;
  auto Signless = SignednessSemantics::Signless;

  nb::class_<DenseElementsAttr, ::mlir::Attribute>(m, "DenseElementsAttr");
  nb::class_<DenseResourceElementsAttr, ::mlir::Attribute>(
      m, "DenseResourceElementsAttr");
  nb::class_<FlatSymbolRefAttr, ::mlir::Attribute>(m, "FlatSymbolRefAttr");
  nb::class_<mlir::AffineMap>(m, "AffineMap");
  nb::class_<mlir::IntegerSet>(m, "IntegerSet");
  nb::class_<mlir::detail::ElementsAttrIndexer>(m, "_ElementsAttrIndexer");

  nb::enum_<SignednessSemantics>(m, "SignednessSemantics")
      .value("signless", SignednessSemantics::Signless)
      .value("signed", SignednessSemantics::Signed)
      .value("unsigned", SignednessSemantics::Unsigned);

  nb::class_<BoolAttr, ::mlir::Attribute>(m, "BoolAttr")
      .def(nb::new_(infer_context(&BoolAttr::get)))
      .def_prop_ro("value", &BoolAttr::getValue);

#include "_core/Graph/BuiltinAttributeInterfaces.cpp.inc"
#include "_core/Graph/BuiltinAttributes.cpp.inc"
#include "_core/Graph/BuiltinDialect.cpp.inc"
#include "_core/Graph/BuiltinLocationAttributes.cpp.inc"
#include "_core/Graph/BuiltinOps.cpp.inc"
#include "_core/Graph/BuiltinTypeInterfaces.cpp.inc"
#include "_core/Graph/BuiltinTypes.cpp.inc"

  nb::cast<nb::class_<::mlir::ModuleOp>>(nb::type<::mlir::ModuleOp>())
      .def(nb::new_(
               [](Location loc, std::optional<StringRef> name = std::nullopt) {
                 return mlir::ModuleOp::create(loc, name);
               }),
           nb::arg("location"), nb::arg("name") = nb::none())
      .def_prop_ro("body", [](mlir::ModuleOp &self) {
        return &self.getBodyRegion().front();
      });

  nb::implicitly_convertible<mlir::Type, mlir::TypeAttr>();
  nb::implicitly_convertible<bool, BoolAttr>();
  nb::implicitly_convertible<std::string, StringAttr>();

  // The default IntegerAttr builders aren't very ergonomic.
  // They work with APInts, but expect you to be able to be passing
  // APInts that are the same width as the IntegerType. There's
  // not really a good way to satisfy this in general right now.
  // Delete these and create a more sensible constructor.
  nb::handle IntegerAttr = nb::type<mlir::IntegerAttr>();
  nb::delattr(IntegerAttr, "__new__");
  nb::delattr(IntegerAttr, "__init__");
  nb::cast<nb::class_<::mlir::IntegerAttr>>(IntegerAttr)
      .def(nb::new_([](mlir::IntegerType type, nb::int_ value) {
             auto str = nb::str(value);
             llvm::APSInt apsint = llvm::APSInt(str.c_str());
             if (type.getWidth() < apsint.getBitWidth() ||
                 (apsint.isNegative() && !type.isSigned())) {
               PyErr_Format(PyExc_ValueError,
                            "IntegerAttr: %R outside range for %R", value.ptr(),
                            nb::cast(type).ptr());
               throw nanobind::python_error();
             }

             RaisingScopedEmitFn emitFn;
             return mlir::IntegerAttr::getChecked(
                 emitFn, type, apsint.extend(type.getWidth()));
           }),
           nb::arg("type"), nb::arg("value") = 0)
      // IndexType is also a valid IntegerAttr type (MLIR's IndexAttr).
      // It has no fixed bit width, so we skip the range check and let MLIR
      // store the value as a 64-bit APInt.
      .def(nb::new_([](mlir::IndexType type, int64_t value) {
             return mlir::IntegerAttr::get(type, value);
           }),
           nb::arg("type"), nb::arg("value") = 0)
      // Single-arg form defaults to IndexType (MLIR's IndexAttr). Convenient
      // shorthand for the common `IntegerAttr(IndexType(), value)` pattern.
      .def(nb::new_([](int64_t value) {
             return mlir::IntegerAttr::get(
                 mlir::IndexType::get(current_context()), value);
           }),
           nb::arg("value"));

  // Register some builtin passes
  nb::module_ passes = m.def_submodule("passes");
  {
    nb::module_ m = passes;

    nb::enum_<mlir::GreedySimplifyRegionLevel>(m, "GreedySimplifyRegionLevel")
        .value("normal", mlir::GreedySimplifyRegionLevel::Normal)
        .value("disabled", mlir::GreedySimplifyRegionLevel::Disabled)
        .value("aggressive", mlir::GreedySimplifyRegionLevel::Aggressive);

    nb::enum_<mlir::ConvergenceFailureAction>(m, "ConvergenceFailureAction")
        .value("warn", mlir::ConvergenceFailureAction::Warn)
        .value("error", mlir::ConvergenceFailureAction::Error)
        .value("silent", mlir::ConvergenceFailureAction::Silent);

#include "_core/Graph/Passes.cpp.inc"
  }
}
