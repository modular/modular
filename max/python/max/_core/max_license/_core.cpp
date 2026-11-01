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

#include "Support/BazelRunfiles.h"
#include "Support/Nanobind/DiscardableAttributes.h"
#include "Support/Nanobind/Pointer.h"
#include "Support/Nanobind/SequenceView.h"
#include "Support/Nanobind/TypeCasters.h"
#include "Support/NixlPluginDir.h"
#include "_core/AsyncRT/Bindings.h"
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "max/c/common.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "nanobind/make_iterator.h"
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/filesystem.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/typing.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Triple.h"
#include <filesystem>
#include <iterator>
#include <optional>
#include <stdlib.h>

namespace nb = nanobind;
using namespace nanobind::literals;
namespace Python = M::Graph::Python;

void pySetEnvIfNotSet(nb::mapping &env, const char *key, const char *value) {
  if (!env.contains(key))
    env[key] = value;
}

std::optional<std::filesystem::path> find_package_root() {
  // If we're in conda, `file` will be something like
  // ...lib/python3.12/site-packages/max
  // and we want
  // ...bin/mojo

  // If we're in pip, `file` will be something like
  // ...max/
  // and we want
  // ...modular/bin/mojo

  // If we're in the open source repo multiple modules are importable as max/
  const auto files = nb::cast<std::vector<std::filesystem::path>>(
      nb::module_::import_("max")
          .attr("__spec__")
          .attr("submodule_search_locations"));

  // Test for an arbitrary file - we assume the package structures for mojo
  // things are otherwise the same.
  for (auto &file : files) {
    const auto maybeCondaPath = file.parent_path()
                                    .parent_path()
                                    .parent_path()
                                    .parent_path()
                                    .parent_path();
    const auto maybeWheelPath = file.parent_path() / "modular";

    // Test for an arbitrary file - we assume the package structures for mojo
    // things are otherwise the same.
    for (const auto &path : {maybeWheelPath, maybeCondaPath})
      if (std::filesystem::exists(path / "bin" / "mojo"))
        return path;
  }

  return std::nullopt;
}

void set_config_override_vars() {
  auto environ =
      nb::borrow<nb::mapping>(nb::module_::import_("os").attr("environ"));
  // suppress transformers logging
  pySetEnvIfNotSet(environ, "TRANSFORMERS_VERBOSITY", "critical");
  pySetEnvIfNotSet(environ, "TRANSFORMERS_NO_ADVISORY_WARNINGS", "1");

  if (llvm::sys::Process::GetEnv("MODULAR_MOJO_MAX_RESPECT_MODULAR_CFG")) {
    // Force respecting the modular.cfg and skip overriding it here.
    return;
  }

  // NOTE: setenv is not threadsafe. However, there are likely
  // no writes happening during this module initialization.
  //
  // We write both libc's environ (so C++ code and subprocess exec() inherit
  // the value) *and* Python's os.environ (so Python-spawned worker
  // subprocesses inherit them; Python caches os.environ and does not
  // refresh from libc on setenv).
  auto setBoth = [&environ](const char *name, const std::string &value) {
    setenv(name, value.c_str(), 0);
    pySetEnvIfNotSet(environ, name, value.c_str());
  };

  // Point NIXL at a resolved per-vendor plugin dir. When that dir is the AMD
  // UCCL flavor, also default the transfer backend to uccl so the backend
  // name the KVTransferEngine requests matches the only plugin staged there
  // (overwrite 0, so an explicit MODULAR_NIXL_TRANSFER_BACKEND still wins).
  auto setPluginDir = [&setBoth](const std::filesystem::path &dir) {
    // setBoth cannot overwrite, so an earlier resolution already owns the var
    // and this call changes nothing — including which libfabric to claim.
    const bool resolvedEarlier =
        llvm::sys::Process::GetEnv("NIXL_PLUGIN_DIR").has_value();
    setBoth("MODULAR_NIXL_PLUGIN_DIR", dir.string());
    setBoth("NIXL_PLUGIN_DIR", dir.string());
    if (dir.filename() == "rocm-uccl")
      setBoth("MODULAR_NIXL_TRANSFER_BACKEND", "uccl");
    // Claim the staged fabric stack here, at max._core import: this is the
    // earliest point the staged copies are known, and they have to beat every
    // other consumer of those SONAMEs in the process — including torch, whose
    // bundled NVSHMEM wants libfabric and whose bundled cuFile RDMA plugin
    // wants libibverbs (see preloadStagedFabricLibs).
    if (!resolvedEarlier)
      M::preloadStagedFabricLibs(dir);
  };

  // NIXL plugin directory for `bazel run` / test workers. Upstream NIXL reads
  // NIXL_PLUGIN_DIR once, at plugin-manager construction, and dlopens the
  // transport plugins from it. Resolve the plugins from the Bazel runfiles and
  // point the var at them. This runs before the package-root early-returns
  // below on purpose: a `bazel run` worker in a source tree has
  // MODULAR_DERIVED_PATH set, which would otherwise skip it. It is a no-op
  // outside Bazel (nothing resolves) and never overrides an already-set value
  // (setenv, overwrite 0), so an installed <root>/lib/nixl layout still wins
  // where it exists (set below). MODULAR_NIXL_PLUGIN_DIR is kept for backwards
  // compatibility.
  if (auto runfilesPlugin = M::findConfigWithRunfiles("nixl_plugin_dir")) {
    // The anchor resolves the cpu-flavor plugin at <base>/cpu/ (staged
    // unconditionally — the GPU flavors are select()ed per vendor), so the
    // vendor-neutral base is two levels up.
    if (auto dir =
            M::resolveNixlPluginDir(std::filesystem::path(*runfilesPlugin)
                                        .parent_path()
                                        .parent_path())) {
      setPluginDir(*dir);
    }
  }

  // Installed conda/venv layout: the NIXL plugins ship in per-vendor
  // subdirectories of <prefix>/lib/nixl. sys.prefix names the active
  // environment root in every install flavor (conda, venv, system python) —
  // including serving containers, where neither VIRTUAL_ENV nor the bin/mojo
  // package-root anchor below resolves. First writer wins (setenv overwrite
  // 0), so a runfiles resolution above still takes precedence.
  const auto sysPrefix =
      nb::cast<std::string>(nb::module_::import_("sys").attr("prefix"));
  if (auto dir = M::resolveNixlPluginDir(std::filesystem::path(sysPrefix) /
                                         "lib" / "nixl")) {
    setPluginDir(*dir);
  }

  std::optional<std::filesystem::path> base = find_package_root();

  if (!base) {
    std::optional<std::string> venvBase =
        llvm::sys::Process::GetEnv("VIRTUAL_ENV");
    if (venvBase) {
      base = *venvBase;
    } else if (llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH")) {
      // Force respecting the modular.cfg and skip overriding it here.
      return;
    }
  }

  if (base) {
    setBoth("MODULAR_MAX_PACKAGE_ROOT", base.value().string());
    setBoth("MODULAR_MOJO_MAX_PACKAGE_ROOT", base.value().string());
    const auto lib = base.value() / "lib";
    setBoth("MODULAR_MOJO_MAX_IMPORT_PATH", (lib / "mojo").string());
    // Installed layout: the NIXL plugins ship in per-vendor subdirectories of
    // <root>/lib/nixl. Only set the var when a vendor directory resolves;
    // otherwise the runfiles value resolved above stands.
    if (auto dir = M::resolveNixlPluginDir(lib / "nixl")) {
      setPluginDir(*dir);
    }
  }
}

NB_MODULE(_core, m) {
  set_config_override_vars();

  m.attr("__version__") = M_version();

  M::AsyncRT::Nanobind::bindMLRT(m.def_submodule("mlrt"));

  Python::bindDriverModule(m.def_submodule("driver"));
  Python::bindDTypeModule(m.def_submodule("dtype"));
  Python::bindEngineModule(m.def_submodule("engine"));
  Python::bindGraphModule(m.def_submodule("graph"));
  Python::bindProfilerModule(m.def_submodule("profiler"));
  Python::bindNIXLModule(m.def_submodule("nixl"));
  Python::bindSafetensorsModule(m.def_submodule("safetensors"));
  Python::bindXXHashModule(m.def_submodule("xxhash"));
  Python::bindXGrammarModule(m.def_submodule("xgrammar"));
  Python::bindMetricsModule(m.def_submodule("metrics"));
  Python::bindRequestContextModule(m.def_submodule("request_context"));
  Python::bindDLPackModule(m.def_submodule("dlpack"));

  // Ensure these types have been loaded
  auto ir = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"));

  // It would be nice if these could be implicitly convertible from
  // `MlirAttribute`, but they can't.
  // - The `Mlir*` type casters upstream can throw unsafely if passed an invalid
  // type.
  // - Since nanobind type casters must be `noexcept`, this can cause a
  // crash during normal overload resolution.
  nb::class_<::mlir::Attribute>(m, "Attribute")
      .def_static("_from_cmlir",
                  static_cast<::mlir::Attribute (*)(MlirAttribute)>(unwrap))
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__hash__",
           [](mlir::Attribute &self) -> nb::int_ {
             return nb::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def_prop_ro("asm",
                   [](mlir::Attribute &self) {
                     std::string result;
                     llvm::raw_string_ostream(result) << self;
                     return result;
                   })
      .def("__repr__",
           [](nb::object self) {
             return nb::str("{}({})").format(self.type().attr("__name__"),
                                             self.attr("asm"));
           })
      // For backwards compatibility with MLIR C API
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, [](mlir::Attribute &self) {
        return nb::steal(mlirPythonAttributeToCapsule(wrap(self)));
      });

  nb::class_<::mlir::NamedAttribute>(
      m, "NamedAttribute",
      nb::sig("class NamedAttribute(tuple[str, Attribute])"))
      .def(nb::init<llvm::StringRef, mlir::Attribute>())
      .def_prop_ro(
          "name",
          [](mlir::NamedAttribute &self) { return self.getName().getValue(); })
      .def_prop_ro("value", &mlir::NamedAttribute::getValue)
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__repr__",
           [](mlir::NamedAttribute &self) {
             return nb::str("NamedAttribute({!r}, {!r})")
                 .format(self.getName().getValue(), self.getValue());
           })
      .def(
          "__iter__",
          [](mlir::NamedAttribute &attr) {
            return nb::iter(nb::make_tuple(attr.getName().getValue(),
                                           attr.getValue()))
                .release();
          },
          nb::sig("def __iter__(self) -> Iterator[str | Attribute]"));

  nb::class_<::mlir::TypeID>(m, "TypeID");

  nb::class_<::mlir::Type>(m, "Type")
      .def_static("_from_cmlir",
                  static_cast<::mlir::Type (*)(MlirType)>(unwrap))
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__hash__",
           [](mlir::Type &self) -> nb::int_ {
             return nb::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def_prop_ro("asm",
                   [](mlir::Type &self) {
                     std::string result;
                     llvm::raw_string_ostream(result) << self;
                     return result;
                   })
      .def("__repr__",
           [](nb::object self) {
             return nb::str("{}({})").format(self.type().attr("__name__"),
                                             self.attr("asm"));
           })
      // For backwards compatibility with MLIR C API
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, [](mlir::Type &self) {
        return nb::steal(mlirPythonTypeToCapsule(wrap(self)));
      });

  nb::dict tBounds;
  tBounds["bound"] = nb::type<mlir::Type>();
  m.attr("T") = nb::type_var("T", **tBounds);

  nb::class_<::mlir::Value>(m, "Value", nb::is_generic(),
                            nb::sig("class Value(typing.Generic[T])"))
      .def_static(
          "_from_cmlir", static_cast<::mlir::Value (*)(MlirValue)>(unwrap),
          nb::sig("def _from_cmlir(value: MlirValue, /) -> Value[Type]"))
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__hash__",
           [](::mlir::Value &self) -> size_t { return mlir::hash_value(self); })
      .def_prop_rw(
          "type", &mlir::Value::getType, &mlir::Value::setType,
          nb::for_getter<nb::sig>("def type(self) -> T"),
          nb::for_setter<nb::sig>("def type(self, type: T, /) -> None"))
      .def_prop_ro(
          "owner",
          [](const mlir::Value &self) {
            if (auto value = mlir::dyn_cast<mlir::OpResult>(self)) {
              return nb::handle(Python::castOp(value.getOwner()));
            }
            return nb::cast(mlir::cast<mlir::BlockArgument>(self).getOwner())
                .release();
          },
          nb::sig("def owner(self) -> Block | Operation"))
      .def_prop_ro("num_uses", &mlir::Value::getNumUses)
      // For backwards compatibility with MLIR C API
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, [](mlir::Value &self) {
        return nb::steal(mlirPythonValueToCapsule(wrap(self)));
      });

  nb::implicitly_convertible<MlirValue, ::mlir::Value>();

  nb::class_<::mlir::OpOperand>(m, "OpOperand")
      .def_prop_rw("value", &mlir::OpOperand::get, &mlir::OpOperand::assign,
                   nb::for_getter<nb::sig>("def value(self) -> Value[Type]"),
                   nb::for_setter<nb::sig>(
                       "def value(self, value: Value[Type], /) -> None"));

  nb::class_<::mlir::OpBuilder::InsertPoint>(m, "InsertPoint");
  nb::class_<::mlir::Block>(m, "Block")
      .def(nb::new_([](std::vector<::mlir::Type> arg_types,
                       std::vector<::mlir::Location> arg_locs) {
             // return a `unique_ptr`; nanobind gives ownership to Python
             // so we don't need to manage block lifetime explicitly.
             auto block = std::make_unique<::mlir::Block>();
             for (auto [t, l] : llvm::zip(arg_types, arg_locs))
               block->addArgument(t, l);
             return block;
           }),
           "arg_types"_a = std::vector<::mlir::Type>{},
           "arg_locs"_a = std::vector<::mlir::Location>{})
      .def_static("_from_cmlir",
                  static_cast<::mlir::Block *(*)(MlirBlock)>(unwrap),
                  nb::rv_policy::reference)
      .def("__len__",
           [](mlir::Block &self) { return self.getOperations().size(); })
      .def(
          "__iter__",
          [](mlir::Block &self) {
            auto ops =
                llvm::map_range(self.getOperations(), [](mlir::Operation &op) {
                  return Python::castOp(&op);
                });
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<mlir::Block>(), "BlockOperations", ops.begin(),
                ops.end());
          },
          nb::sig("def __iter__(self) -> Iterable[Operation]"))
      .def(
          "__getitem__",
          [](mlir::Block &self, ssize_t index) {
            auto &ops = self.getOperations();
            if (index >= 0) {
              auto it = ops.begin();
              while (index-- > 0)
                ++it;
              return Python::castOp(&*it);
            } else {
              auto it = ops.rbegin();
              while (index++ < -1)
                ++it;
              return Python::castOp(&*it);
            }
          },
          nb::sig("def __getitem__(self, index: int, /) -> Operation"))
      .def_prop_ro("arguments", &mlir::Block::getArguments)
      .def_prop_ro(
          "parent_op",
          [](mlir::Block &self) { return Python::castOp(self.getParentOp()); })
      .def("add_argument", &mlir::Block::addArgument, nb::keep_alive<1, 0>())
      .def("erase_argument", &mlir::Block::eraseArgument)
      .def_prop_ro("end",
                   [](nb::object self) {
                     mlir::Block *block = nb::cast<mlir::Block *>(self);
                     return ::mlir::OpBuilder::InsertPoint(block, block->end());
                   })
      // For backwards compatibility with MLIR C API
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, [](::mlir::Block &self) {
        return nb::steal(mlirPythonBlockToCapsule(wrap(&self)));
      });

  // We don't really want nb::bind_map.
  //   - Users can't meaningfully create these
  //   - bind_map expects a reference-semantic operator[], which we don't have
  nb::class_<Python::DiscardableAttributes>(m, "DiscardableAttributes")
      .def("__getitem__",
           [](Python::DiscardableAttributes &self, llvm::StringRef name) {
             if (auto attr = self.op->getDiscardableAttr(name)) {
               return attr;
             }
             throw nb::key_error("No such attribute present");
           })
      .def("__setitem__",
           [](Python::DiscardableAttributes &self, llvm::StringRef name,
              mlir::Attribute attr) {
             return self.op->setDiscardableAttr(name, attr);
           })
      .def("__delitem__",
           [](Python::DiscardableAttributes &self, llvm::StringRef name) {
             return self.op->removeDiscardableAttr(name);
           })
      .def("__contains__",
           [](Python::DiscardableAttributes &self, llvm::StringRef name) {
             return bool(self.op->getDiscardableAttr(name));
           })
      .def("__len__",
           [](Python::DiscardableAttributes &self) {
             auto range = self.op->getDiscardableAttrs();
             return std::distance(range.begin(), range.end());
           })
      .def("__bool__",
           [](Python::DiscardableAttributes &self) {
             return !self.op->getDiscardableAttrs().empty();
           })
      .def("keys",
           [](Python::DiscardableAttributes &self) {
             auto range = llvm::map_range(self.op->getDiscardableAttrs(),
                                          [](const mlir::NamedAttribute &attr) {
                                            return attr.getName().getValue();
                                          });
             return nb::make_iterator<nb::rv_policy::reference_internal>(
                 nb::type<Python::DiscardableAttributes>(),
                 "DiscardableAttributesKeys", range.begin(), range.end());
           })
      .def("items",
           [](Python::DiscardableAttributes &self) {
             auto range = self.op->getDiscardableAttrs();
             return nb::make_iterator<nb::rv_policy::reference_internal>(
                 nb::type<Python::DiscardableAttributes>(),
                 "DiscardableAttributesItems", range.begin(), range.end());
           })
      .def("values",
           [](Python::DiscardableAttributes &self) {
             auto range = llvm::map_range(self.op->getDiscardableAttrs(),
                                          [](const mlir::NamedAttribute &attr) {
                                            return attr.getValue();
                                          });
             return nb::make_iterator<nb::rv_policy::reference_internal>(
                 nb::type<Python::DiscardableAttributes>(),
                 "DiscardableAttributesValues", range.begin(), range.end());
           })
      .def("__iter__",
           [](nb::handle self) {
             return nb::cast<nb::typed<nb::iterator, nb::str>>(
                 self.attr("keys")());
           })
      .def("__repr__", [](Python::DiscardableAttributes &self) {
        nb::dict dict;
        for (auto namedAttribute : self.op->getDiscardableAttrs()) {
          dict[nb::cast(namedAttribute.getName().getValue())] =
              namedAttribute.getValue();
        }
        return nb::str("discardable_attributes({})").format(dict);
      });

  nb::class_<::mlir::OpState>(m, "Operation")
      .def_static("_from_cmlir",
                  [](nb::handle op) {
                    nb::handle capsule =
                        nb::getattr(op, MLIR_PYTHON_CAPI_PTR_ATTR).release();
                    MlirOperation cOp =
                        mlirPythonCapsuleToOperation(capsule.ptr());
                    return *(mlir::OpState *)&cOp;
                  })
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def_prop_ro(
          "context",
          [](mlir::OpState &self) { return self.getOperation()->getContext(); })
      .def_prop_ro("operands",
                   [](mlir::OpState &self) {
                     return self.getOperation()->getOpOperands();
                   })
      .def_prop_ro(
          "results",
          [](mlir::OpState &self) { return self.getOperation()->getResults(); })
      .def_prop_ro(
          "regions",
          [](mlir::OpState &self) { return self.getOperation()->getRegions(); })
      .def_prop_ro("parent_op",
                   [](mlir::OpState &self) {
                     return Python::castOp(self.getOperation()->getParentOp());
                   })
      .def(
          "verify",
          [](mlir::OpState &self, bool verify_recursively) {
            M::Graph::Python::RaisingScopedEmitFn unusedEmitFn(
                self->getContext());
            // Intentionally ignoring the result, if it fails it will propagate
            // as an error via the EmitFn above
            [[maybe_unused]] auto result =
                mlir::verify(self.getOperation(), verify_recursively);
          },
          nb::arg("verify_recursively") = true)
      .def("move_after",
           [](mlir::OpState &self, mlir::OpState &other) {
             self.getOperation()->moveAfter(other.getOperation());
           })
      // Like parse_module, the returned op is detached and its lifetime
      // is managed by the Python wrapper.
      .def("clone",
           [](mlir::OpState &self) {
             return Python::castOp(self.getOperation()->clone());
           })
      .def_prop_ro("discardable_attributes",
                   [](mlir::OpState &self) {
                     return Python::DiscardableAttributes(self.getOperation());
                   })
      // Returns the operation's MLIR assembly text. Mirrors the cmlir
      // ``Operation.get_asm`` kwargs so callers that need explicit
      // ``OpPrintingFlags`` (for example, a stable hash basis that strips
      // debug info and uses local SSA scope) don't have to round-trip
      // through the cmlir bindings. Default-flag callers can simply use
      // ``op.asm()`` instead of ``str(op)``.
      .def(
          "asm",
          [](mlir::OpState &self, bool enable_debug_info,
             bool pretty_debug_info, bool print_generic_op_form,
             bool use_local_scope, bool assume_verified, bool skip_regions) {
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo(enable_debug_info, pretty_debug_info);
            flags.printGenericOpForm(print_generic_op_form);
            flags.useLocalScope(use_local_scope);
            flags.assumeVerified(assume_verified);
            flags.skipRegions(skip_regions);
            std::string result;
            llvm::raw_string_ostream os(result);
            self.getOperation()->print(os, flags);
            return result;
          },
          nb::kw_only(), "enable_debug_info"_a = false,
          "pretty_debug_info"_a = false, "print_generic_op_form"_a = false,
          "use_local_scope"_a = false, "assume_verified"_a = false,
          "skip_regions"_a = false)
      // Serialize to MLIR bytecode. Unlike ``asm`` (textual), bytecode is a
      // lossless transport, so it round-trips IR that the textual parser
      // mishandles (for example f64 ``#M.dense_array`` literals). Pair with
      // ``from_bytecode`` to clone an op into a different MLIRContext.
      .def_prop_ro("bytecode",
                   [](mlir::OpState &self) {
                     std::string result;
                     llvm::raw_string_ostream os(result);
                     if (mlir::failed(mlir::writeBytecodeToFile(
                             self.getOperation(), os)))
                       throw std::runtime_error(
                           "Failed to serialize operation to bytecode");
                     return nb::bytes(result.data(), result.size());
                   })
      // Parse MLIR bytecode (as produced by ``bytecode``) into a detached
      // ``builtin.ModuleOp`` owned by the returned Python wrapper. The module
      // is materialized in ``context``, so this is the lossless way to move a
      // module between MLIRContexts.
      .def_static(
          "from_bytecode",
          [](nb::bytes data, mlir::MLIRContext *context) -> mlir::ModuleOp {
            llvm::StringRef bytes(data.c_str(), data.size());
            mlir::OwningOpRef<mlir::ModuleOp> module =
                mlir::parseSourceString<mlir::ModuleOp>(bytes, context);
            if (!module)
              throw std::invalid_argument(
                  "Failed to parse MLIR module from bytecode");
            return module.release();
          },
          "data"_a, "context"_a, nb::keep_alive<0, 2>())
      .def("__repr__",
           [](nb::object self) {
             return nb::str("{}({})").format(self.type().attr("__name__"),
                                             self.attr("asm")());
           })
      // For backwards compatibility with MLIR C API
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, [](mlir::OpState &self) {
        return nb::steal(
            mlirPythonOperationToCapsule(wrap(self.getOperation())));
      });

  nb::class_<::mlir::OpBuilder>(m, "OpBuilder")
      .def(nb::new_([](::mlir::OpBuilder::InsertPoint &ip) {
             return std::make_unique<::mlir::OpBuilder>(ip.getBlock(),
                                                        ip.getPoint());
           }),
           nb::rv_policy::reference_internal);

  nb::class_<::mlir::Region>(m, "Region")
      .def_prop_ro("front", &mlir::Region::front)
      .def_prop_ro("back", &mlir::Region::back)
      .def(
          "append",
          [](::mlir::Region &self, nb::object block_obj) {
            // Take a raw pointer view (does not relinquish the wrapper).
            auto *block = nb::cast<::mlir::Block *>(block_obj);
            // `Region::push_back` will unsafely take ownership.
            // Defensively check and reject blocks owned by regions already.
            if (block->getParent())
              throw std::invalid_argument(
                  "Region.append: block is already attached to a region");
            self.push_back(block);
            // Region's intrusive block list now owns the block. Mark the
            // Python wrapper as a non-owning view so GC won't delete it.
            // The `keep_alive<2, 1>` below ties the region's lifetime to
            // this wrapper, so the wrapper stays valid even after callers
            // drop their Python references to the region / parent op.
            nb::inst_set_state(block_obj, /*ready=*/true, /*destruct=*/false);
          },
          nb::keep_alive<2, 1>(),
          nb::sig("def append(self, block: Block, /) -> None"));

  nb::class_<::mlir::LocationAttr>(m, "LocationAttr");

  nb::class_<::llvm::BitVector>(m, "_BitVector");
  nb::class_<::llvm::Triple>(m, "_TargetTriple");
  nb::enum_<::llvm::Reloc::Model>(m, "_RelocationModel");
  nb::class_<::mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>(
      m, "_MemoryEffect");

  nb::class_<Python::SequenceView>(m, "SequenceView")
      .def("__len__", [](Python::SequenceView &self) { return self.len; })
      .def(
          "__getitem__",
          [](Python::SequenceView &self, size_t item) {
            if (item < 0)
              item += self.len;
            if (item >= self.len)
              throw nb::index_error();
            return self.from_cpp(self[item], nb::rv_policy::reference, nullptr);
          },
          nb::keep_alive<1, 0>())
      .def("__repr__", [](nb::handle self) {
        return nb::str("sequence_view({})").format(nb::repr(nb::list(self)));
      });

  nb::class_<Python::Pass>(m, "Pass")
      .def_prop_ro("name",
                   [](Python::Pass &self) { return self.pass->getName(); })
      .def_prop_ro("op_name",
                   [](Python::Pass &self) { return self.pass->getOpName(); })
      .def("__repr__", [](Python::Pass &self) {
        // Should also indicate which ops this pass runs on, and any configured
        // options.
        return nb::str("{}()").format(self.pass->getName());
      });

  nb::class_<mlir::OpPassManager>(m, "OpPassManager");

  m.def("lower", [](mlir::ModuleOp module,
                    std::vector<std::unique_ptr<Python::Pass>> passes) {
    auto ctx = module.getContext();
    auto passManager = mlir::PassManager::on<mlir::ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);
    for (auto &&pass : passes)
      passManager.addPass(std::move(pass->pass));
    M::Graph::Python::RaisingScopedEmitFn emitFn(ctx);
    return passManager.run(module).succeeded();
  });

  // Parse a serialized MLIR module's textual form into a typed
  // ``builtin.ModuleOp``. Mirrors cmlir's :meth:`mlir.Module.parse` but stays
  // inside the nanobind binding world, so callers don't have to bridge a
  // cmlir Python wrapper just to keep its underlying lifetime alive.
  //
  // Mishandles float attributes, so prefer cmlir's ``mlir.Module.parse`` for
  // any text that might carry one: this extension static-links its own LLVM,
  // and matching APFloat semantics by pointer against its own copy of the
  // ``APFloatBase::sem*`` statics turns f64 literals into denormals and aborts
  // on hex f32 ones. ``max.graph.graph._parse_module_via_cmlir`` parses in
  // cmlir and hands the module back over as bytecode. See GEX-4052.
  m.def(
      "parse_module",
      [](const std::string &asm_str,
         mlir::MLIRContext *context) -> mlir::ModuleOp {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(asm_str, context);
        if (!module)
          throw std::invalid_argument("Failed to parse MLIR module");
        return module.release();
      },
      "asm"_a, "context"_a);

  auto dialects = m.def_submodule("dialects");
  Python::bindBuiltinDialect(dialects.def_submodule("builtin"));
  Python::bindMDialect(dialects.def_submodule("m"));
  Python::bindKGENDialect(dialects.def_submodule("kgen"));
  Python::bindMOSHDialect(dialects.def_submodule("mosh"));
  Python::bindMODialect(dialects.def_submodule("mo"));
  Python::bindRMODialect(dialects.def_submodule("rmo"));
}
