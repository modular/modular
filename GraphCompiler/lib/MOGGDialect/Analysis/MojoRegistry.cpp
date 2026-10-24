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

#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/AssertStream.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

#include <tuple>

using namespace M;
using namespace M::MOGG;

MojoRegistry::MojoRegistry(mlir::ModuleOp module) : deviceInfoMapping(module) {
  for (auto decl : module.getOps<KernelDeclOp>()) {
    kernelByName[decl.getRegistration()].push_back(decl);
    kernelBySymName[decl.getSymName()] = decl;
  }
  for (auto decl : module.getOps<PrimitiveDeclOp>()) {
    llvm::StringRef reg = decl.getRegistration();
    if (!primitiveByName.contains(reg))
      primitiveByName[reg] = decl;
  }
  for (auto decl : module.getOps<ShapeFunctionDeclOp>()) {
    shapeFuncByName[decl.getRegistration()] = decl;
    shapeFuncBySymName[decl.getSymName()] = decl;
  }
}

llvm::ArrayRef<KernelDeclOp>
MojoRegistry::lookupKernel(llvm::StringRef name) const {
  auto it = kernelByName.find(name);
  if (it == kernelByName.end())
    return {};
  return it->second;
}

bool MojoRegistry::containsKernel(llvm::StringRef name) const {
  return kernelByName.contains(name);
}

std::optional<llvm::StringRef>
MojoRegistry::suggestKernelName(llvm::StringRef name) const {
  // A typo changes about one character in three, and at least a couple.
  unsigned maxDistance = std::max<size_t>(2, name.size() / 3);
  std::optional<llvm::StringRef> best;
  unsigned bestDistance = maxDistance + 1;
  for (const auto &entry : kernelByName) {
    llvm::StringRef candidate = entry.getKey();
    unsigned distance =
        name.edit_distance(candidate, /*AllowReplacements=*/true,
                           /*MaxEditDistance=*/maxDistance);
    // Ties break by name, so the suggestion doesn't depend on hash order.
    if (distance < bestDistance ||
        (distance == bestDistance && best && candidate < *best)) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return best;
}

KernelDeclOp
MojoRegistry::lookupKernelBySymName(llvm::StringRef symName) const {
  KernelDeclOp decl = lookupKernelBySymNameOrNull(symName);
  ASSERT_STREAM(decl, "no kernel registered for '" << symName << "'");
  return decl;
}

KernelDeclOp
MojoRegistry::lookupKernelBySymNameOrNull(llvm::StringRef symName) const {
  return kernelBySymName.lookup(symName);
}

PrimitiveDeclOp MojoRegistry::lookupPrimitive(llvm::StringRef name) const {
  auto it = primitiveByName.find(name);
  ASSERT_STREAM(it != primitiveByName.end(),
                "no primitive registered for '" << name << "'");
  return it->second;
}

bool MojoRegistry::containsPrimitive(llvm::StringRef name) const {
  return primitiveByName.contains(name);
}

ShapeFunctionDeclOp
MojoRegistry::lookupShapeFunction(llvm::StringRef name) const {
  auto it = shapeFuncByName.find(name);
  if (it == shapeFuncByName.end())
    return {};
  return it->second;
}

ShapeFunctionDeclOp
MojoRegistry::lookupShapeFunctionBySymName(llvm::StringRef symName) const {
  ShapeFunctionDeclOp decl = lookupShapeFunctionBySymNameOrNull(symName);
  ASSERT_STREAM(decl, "no shape function registered for sym_name '" << symName
                                                                    << "'");
  return decl;
}

ShapeFunctionDeclOp MojoRegistry::lookupShapeFunctionBySymNameOrNull(
    llvm::StringRef symName) const {
  return shapeFuncBySymName.lookup(symName);
}

bool MojoRegistry::containsShapeFunction(llvm::StringRef name) const {
  return shapeFuncByName.contains(name);
}

static bool deviceInfoMatches(DeviceInfoAttr kernelInfo,
                              DeviceInfoAttr deviceInfo) {
  auto matches = [](llvm::StringRef kernelField, llvm::StringRef runtimeField) {
    return kernelField.empty() || kernelField == runtimeField;
  };
  return matches(kernelInfo.getLabel(), deviceInfo.getLabel()) &&
         matches(kernelInfo.getApi(), deviceInfo.getApi()) &&
         matches(kernelInfo.getArch(), deviceInfo.getArch()) &&
         matches(kernelInfo.getModel(), deviceInfo.getModel());
}

static int deviceInfoSpecificity(DeviceInfoAttr infos) {
  return !infos.getLabel().empty() + !infos.getApi().empty() +
         !infos.getArch().empty() + !infos.getModel().empty();
}

// Whether the kernel's `execute` (or `execute_device`, for a declarative
// kernel) declares a tile-based fused output (_FusedComputeOutputTileTensor or
// _FusedOutputTileTensor), which only lowers on a tile_based_fusion device.
// Transitional, remove per GEX-3941.
static bool kernelDeclaresTileOutput(KernelDeclOp decl) {
  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction.has_value())
    return false;
  for (const auto &arg : adaptor.executeFunction->outputArguments) {
    const auto *tensorArg =
        std::get_if<KernelDeclTensorAdaptor>(&arg.underlyingType);
    if (tensorArg &&
        (tensorArg->ioSpec == IOSpec::FusedComputeOutputTileTensor ||
         tensorArg->ioSpec == IOSpec::FusedOutputTileTensor))
      return true;
  }
  return false;
}

// The order candidates win in, compared lexicographically: a user-defined
// kernel beats a built-in one, then more matching device fields beat fewer.
static std::tuple<bool, int> resolutionRank(KernelDeclOp decl) {
  return {decl.getIsUserDefined(),
          deviceInfoSpecificity(decl.getRegistrationDeviceInfo())};
}

ErrorOr<KernelDeclOp>
MojoRegistry::getResolvedKernel(llvm::StringRef name,
                                DeviceRefAttr device) const {
  DeviceInfoAttr deviceInfo = deviceInfoMapping.get(device);
  auto it = kernelByName.find(name);
  ASSERT_STREAM(it != kernelByName.end(),
                "no kernel registered for '" << name << "'");

  llvm::SmallVector<KernelDeclOp> candidates;
  for (KernelDeclOp decl : it->second) {
    if (!deviceInfoMatches(decl.getRegistrationDeviceInfo(), deviceInfo))
      continue;
    // TODO(GEX-3941): on non-tile devices, skip tile-output candidates so the
    // generic (SIMD) registration is selected.
    if (!deviceInfo.getTileBasedFusion() && kernelDeclaresTileOutput(decl))
      continue;
    candidates.push_back(decl);
  }

  if (candidates.empty()) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "no kernel matching device info for '" << name << "' on device "
       << mlir::Attribute(deviceInfo) << "; candidates:";
    for (KernelDeclOp decl : it->second)
      os << "\n  - " << mlir::Attribute(decl.getRegistrationDeviceInfo())
         << " (" << decl.getSymName() << ")";
    return Error(os.str());
  }

  KernelDeclOp best =
      *llvm::max_element(candidates, [](KernelDeclOp a, KernelDeclOp b) {
        return resolutionRank(a) < resolutionRank(b);
      });
  ASSERT_STREAM(llvm::count_if(candidates,
                               [&](KernelDeclOp decl) {
                                 return resolutionRank(decl) ==
                                        resolutionRank(best);
                               }) == 1,
                "ambiguous kernel resolution for '"
                    << name << "': multiple "
                    << (best.getIsUserDefined() ? "user-defined" : "built-in")
                    << " registrations match the runtime device with equal "
                       "specificity");
  return best;
}

DeviceRefAttr MojoRegistry::getDeviceRefForOp(mlir::Operation *op) {
  if (auto hasDevice = mlir::dyn_cast<MO::MOHasDeviceInterface>(op)) {
    auto devices = hasDevice.getExecutionDevices();
    if (!devices.empty())
      return devices[0];
  }
  return getHostDevice(op->getContext());
}

void M::MOGG::copyKernelDecls(mlir::ModuleOp src, mlir::ModuleOp dst) {
  // Idempotent: a mapping attribute means `dst` is already seeded (e.g. by a
  // prior compile, or in re-imported textual MLIR), and re-cloning the decls
  // would redefine their symbols.
  if (dst->hasAttr(kOpaqueTypeMappingAttrName))
    return;

  mlir::OpBuilder builder(dst.getContext());
  builder.setInsertionPointToEnd(dst.getBody());
  for (auto decl : src.getOps<KernelDeclOp>())
    builder.clone(*decl.getOperation());
  for (auto decl : src.getOps<PrimitiveDeclOp>())
    builder.clone(*decl.getOperation());
  for (auto decl : src.getOps<ShapeFunctionDeclOp>())
    builder.clone(*decl.getOperation());

  // Also carry over the opaque-type mapping so `mogg-rewrite-opaque-types`
  // can run on `dst` without re-loading any packages.
  if (auto mapping = src->getAttrOfType<mlir::DictionaryAttr>(
          kOpaqueTypeMappingAttrName)) {
    dst->setAttr(kOpaqueTypeMappingAttrName, mapping);
  }
}

namespace {

/// Owns the parser state needed to resolve module names and walk the module
/// import directories. One instance serves any number of resolutions, so
/// callers pay the `SharedState` construction cost (config read, dialect
/// preloading) once.
class LITModuleResolver {
public:
  explicit LITModuleResolver(mlir::MLIRContext *ctx)
      : config(ctx, options), sharedState(sourceMgr, config) {}

  /// Resolve `moduleName` to the path of its precompiled binary package.
  ErrorOr<std::string> resolveBinaryPackagePath(llvm::StringRef moduleName) {
    auto modulePath = sharedState.getModuleLoader().resolveModulePath(
        moduleName, llvm::SMLoc());
    if (!modulePath)
      return Error(
          llvm::formatv("Failed to resolve module path for `{}`", moduleName)
              .str());
    if (!modulePath->isPrecompiled()) {
      return Error(
          llvm::formatv("Resolved module path for `{}` to source directory "
                        "`{}` but a .mojoc precompiled file was expected "
                        "instead",
                        moduleName, modulePath->path)
              .str());
    }
    return modulePath->path;
  }

  KGEN::LIT::SharedState &getSharedState() { return sharedState; }

private:
  // Order matters: `config` refers to `options`, `sharedState` to both.
  llvm::SourceMgr sourceMgr;
  KGEN::CompilationOptions options;
  KGEN::LIT::ParserConfig config;
  KGEN::LIT::SharedState sharedState;
};

ErrorOr<llvm::SmallVector<std::string>>
resolveBuiltInKernelPackagePaths(LITModuleResolver &resolver) {
  llvm::SmallVector<std::string> paths;
  for (llvm::StringRef moduleName : kMoggRequiredKernelModuleNames) {
    auto pathOr = resolver.resolveBinaryPackagePath(moduleName);
    if (pathOr.isError())
      return pathOr.takeError();
    paths.push_back(*pathOr);
  }

  // Skipped rather than fatal: a tree without these still has a complete
  // registry of the open-source kernels, and a graph that uses one of their
  // ops fails later with that op unresolved rather than here with every op
  // unresolved.
  for (llvm::StringRef moduleName : kMoggOptionalKernelModuleNames) {
    auto pathOr = resolver.resolveBinaryPackagePath(moduleName);
    if (!pathOr.isError())
      paths.push_back(*pathOr);
  }

  return paths;
}

} // namespace

ErrorOr<llvm::SmallVector<std::string>>
M::MOGG::getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx) {
  LITModuleResolver resolver(ctx);
  return resolveBuiltInKernelPackagePaths(resolver);
}

ErrorOr<llvm::SmallVector<std::string>>
M::MOGG::getKernelPackagePathsForCacheKey(mlir::MLIRContext *ctx) {
  LITModuleResolver resolver(ctx);

  // The built-in packages are required; fail loudly if they don't resolve.
  auto pathsOr = resolveBuiltInKernelPackagePaths(resolver);
  if (pathsOr.isError())
    return pathsOr.takeError();
  llvm::SmallVector<std::string> paths = std::move(*pathsOr);

  // Collect every binary package visible on the import path; the header
  // comment explains why their contents must reach the cache key.
  llvm::StringSet<> seen;
  resolver.getSharedState().getModuleLoader().traverseImportDirectories(
      /*importBufferFileId=*/0, [&](llvm::StringRef dir) {
        std::error_code ec;
        for (llvm::sys::fs::directory_iterator it(dir, ec), e; it != e && !ec;
             it.increment(ec)) {
          const std::string &path = it->path();
          if (M::Filesystem::isMojoBinaryPackagePath(path))
            seen.insert(path);
        }
        return mlir::WalkResult::advance();
      });
  // Append the collected packages, minus the built-ins already in `paths`,
  // sorted for a deterministic order independent of directory iteration.
  llvm::SmallVector<std::string> found;
  for (const auto &entry : seen)
    if (!llvm::is_contained(paths, entry.getKey()))
      found.push_back(entry.getKey().str());
  llvm::sort(found);
  llvm::append_range(paths, found);
  return paths;
}

namespace M::MOGG {

// ---------------------------------------------------------------------------
// Custom op parameter validation using KernelDeclOp
// ---------------------------------------------------------------------------

static bool isStaticStringLoweredPattern(Type ty) {
  auto structTy = dyn_cast<KGEN::StructType>(ty);
  if (!structTy || structTy.getIsParamPack())
    return false;

  std::optional<SmallVector<Type>> elementTypes = structTy.getElementTypes();
  if (!elementTypes || elementTypes->size() != 2)
    return false;

  auto ptrTy = dyn_cast<KGEN::PointerType>((*elementTypes)[0]);
  if (!ptrTy || !isa<KGEN::NoneType>(ptrTy.getElementType()))
    return false;

  return isa<IndexType>((*elementTypes)[1]);
}

static bool isLITStaticStringType(Type ty) {
  auto structTy = KGEN::sugarDynCast<KGEN::LIT::StructType>(ty);
  if (!structTy) {
    if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(ty)) {
      structTy =
          KGEN::sugarDynCast<KGEN::LIT::StructType>(ref.getElementType());
    }
  }
  if (!structTy)
    return false;

  auto symbol = structTy.getSymbol();
  std::string symbolName =
      llvm::formatv("{}::{}", symbol.getRootReference().strref(),
                    symbol.getLeafReference().strref())
          .str();
  if (symbolName != kMojoInternalStaticStringTypeName)
    return false;

  auto paramValues = structTy.getParamValues();
  if (paramValues.empty())
    return false;

  TypedAttr originAttr = paramValues[0];
  if (KGEN::sugarIsa<KGEN::ParamDeclRefAttr>(originAttr))
    return false;

  return true;
}

static StringAttr maybeDemangleName(StringAttr name) {
  StringRef demangled = KGEN::demangleParameterName(name);
  if (demangled != name)
    return StringAttr::get(name.getContext(), demangled);
  return name;
}

/// Return the "root::leaf" name of a parameter's declared struct type, or
/// null for non-structs. Distinguishes e.g. `std::StaticString` from
/// `std::StringSpan`, which the bare KGEN type does not.
static StringAttr getParamTypeName(MLIRContext *ctx, Type paramType) {
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(paramType))
    paramType = ref.getElementType();
  auto structTy = KGEN::sugarDynCast<KGEN::LIT::StructType>(paramType);
  if (!structTy)
    return nullptr;
  auto sym = structTy.getSymbol();
  return StringAttr::get(ctx, llvm::formatv("{}::{}",
                                            sym.getRootReference().strref(),
                                            sym.getLeafReference().strref())
                                  .str());
}

static ErrorOrSuccess validateCustomOpParameters(const MojoRegistry &registry,
                                                 MO::CustomOp customOp) {
  if (!registry.containsKernel(customOp.getSymbol())) {
    return Error(
        llvm::formatv("no kernel registered for '{}'", customOp.getSymbol())
            .str());
  }
  auto declOpOr =
      registry.getResolvedKernel(customOp.getSymbol(), customOp.getDevice());
  if (declOpOr.isError())
    return declOpOr.takeError();
  KernelDeclOp declOp = *declOpOr;
  // The user-supplied `parameters` dict on `mo.custom` can target struct-level
  // compile-time params (e.g. `[interleaved: Bool]`) or any fn-level params
  // declared on the registered functions. Collect struct params first to
  // mirror Mojo's merged `@register` parameter layout, then union the param
  // lists from each function variant (some kernels declare params only on
  // `elementwise` instead of `execute`).
  SmallVector<KGEN::ParamDeclAttr> inputParamsVec;
  llvm::SmallDenseSet<StringRef> seen;
  auto append = [&](KGEN::ParamDeclAttr p) {
    if (seen.insert(p.getName().strref()).second)
      inputParamsVec.push_back(p);
  };
  for (auto a : declOp.getStructParams())
    append(cast<KGEN::ParamDeclAttr>(a));
  auto appendFn = [&](std::optional<MOGG::KernelFunctionAttr> fn) {
    if (!fn)
      return;
    for (auto a : fn->getParams())
      append(cast<KGEN::ParamDeclAttr>(a));
  };
  appendFn(declOp.getExecute());
  appendFn(declOp.getElementwise());
  if (registry.containsShapeFunction(declOp.getRegistration()))
    appendFn(
        registry.lookupShapeFunction(declOp.getRegistration()).getFunction());
  ArrayRef<KGEN::ParamDeclAttr> inputParams = inputParamsVec;

  struct MojoParameterAdaptor {
    KGEN::ParamDeclAttr attr;
    StringAttr typeName;
  };
  llvm::DenseMap<StringAttr, MojoParameterAdaptor> paramDecls;

  MLIRContext *ctx = customOp.getContext();
  for (auto param : inputParams)
    paramDecls[maybeDemangleName(param.getName())] = {
        param, getParamTypeName(ctx, param.getType())};

  for (const NamedAttribute &namedAttr : customOp.getParameters()) {
    auto it = paramDecls.find(namedAttr.getName());
    if (it == paramDecls.end()) {
      return Error(llvm::formatv(
          "parameter '{}' supplied to custom op '{}' does not exist in the "
          "corresponding Mojo code",
          namedAttr.getName().strref(), customOp.getSymbol()));
    }

    auto [customParamDecl, typeName] = it->second;
    Attribute mlirParamAttr = namedAttr.getValue();

    Type kgenParamTy = customParamDecl.getType();
    auto asIntegerType = dyn_cast<IntegerType>(kgenParamTy);

    if (isa<IndexType>(kgenParamTy)) {
      auto asInt = dyn_cast<IntegerAttr>(mlirParamAttr);
      if (asInt == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an integer type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (asIntegerType && asIntegerType.getWidth() == 1) {
      auto asBool = dyn_cast<BoolAttr>(mlirParamAttr);
      if (asBool == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an boolean type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (isa<KGEN::StringType>(kgenParamTy)) {
      if (typeName && typeName == MOGG::kMojoInternalStaticStringTypeName) {
        [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        ASSERT_STREAM(asString != nullptr,
                      "failed to convert custom op attribute "
                          << mlirParamAttr
                          << " to expected mojo kernel param type "
                          << kgenParamTy << ": expected string type");
      } else if (typeName) {
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses a string-like type that is "
            "not supported. Use type StaticString instead to define a string "
            "parameter for a custom op.",
            customOp.getSymbol(), customParamDecl.getName()));
      } else {
        [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        ASSERT_STREAM(asString != nullptr,
                      "failed to convert custom op attribute "
                          << mlirParamAttr
                          << " to expected mojo kernel param type "
                          << kgenParamTy << ": expected string type");
      }
    } else if (isa<KGEN::DTypeType>(kgenParamTy)) {
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoDtypeTypeName) {
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoStaticStringTypeName) {
      auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName == MOGG::kMojoInternalStaticStringTypeName) {
      bool isStaticString = isLITStaticStringType(kgenParamTy) ||
                            isStaticStringLoweredPattern(kgenParamTy);
      if (isStaticString) {
        auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        if (asString == nullptr) {
          return Error(llvm::formatv(
              "failed to convert custom op '{}' attribute '{}' with value "
              "'{}' to the proper Mojo parameter type: expected string type",
              customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
        }
      } else {
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses type StringSpan. Use type "
            "StaticString instead to define a string parameter for a custom "
            "op.",
            customOp.getSymbol(), customParamDecl.getName()));
      }
    } else if (typeName && (typeName == MOGG::kMojoInternalDpsSimdTypeName)) {
      // `Float32` (SIMD[f32, 1]) params arrive as the same rank-0 f32
      // elements attribute the RMS norm ops use for their scalar
      // attributes (see `MO_F32ScalarAttr`); integer SIMD params arrive
      // as `IntegerAttr`. Only f32 matches here: `MO_F32ScalarAttr` and
      // the Python side (`rms_norm._float32_attr`) are f32-only. A float
      // value against an integer SIMD param is not caught here — this
      // branch keys on the SIMD kind, not the element type — and fails
      // later, at Mojo elaboration.
      if (auto elems = dyn_cast<ElementsAttr>(mlirParamAttr);
          elems && elems.getShapedType().getRank() == 0 &&
          elems.getElementType().isF32())
        continue;
      auto asInt = dyn_cast<IntegerAttr>(mlirParamAttr);
      if (asInt == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an integer type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoInternalDpsBoolTypeName) {
      auto asBool = dyn_cast<BoolAttr>(mlirParamAttr);
      if (asBool == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a boolean type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               (typeName == MOGG::kMojoInternalStringSliceTypeName ||
                typeName == MOGG::kMojoStringTypeName)) {
      std::string userTypeName =
          typeName == MOGG::kMojoInternalStringSliceTypeName ? "StringSpan"
                                                             : "String";
      return Error(llvm::formatv(
          "in custom op '{}' attribute '{}' uses type {}. Use type "
          "StaticString instead to define a string parameter for a custom op.",
          customOp.getSymbol(), customParamDecl.getName(), userTypeName));
    } else if (!typeName && isStaticStringLoweredPattern(kgenParamTy)) {
      auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else {
      return Error(llvm::formatv(
          "failed to convert custom op '{}' attribute '{}' with value '{}' to "
          "the proper Mojo parameter type: no type conversion found to type "
          "'{}'",
          customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr,
          kgenParamTy));
    }
  }

  return success();
}

} // namespace M::MOGG

/// Formats `message` as a Mojo-style diagnostic at `loc`, with the Mojo
/// source line, and a note at `noteLoc` when it points elsewhere.
static Error kernelError(Location loc, const std::string &message,
                         std::optional<Location> noteLoc = std::nullopt,
                         const std::string &note = "") {
  mlir::Diagnostic diag(loc, mlir::DiagnosticSeverity::Error);
  diag << message;
  if (noteLoc && *noteLoc != loc)
    diag.attachNote(*noteLoc) << note;
  return Error(llvm::formatv(
      "\n{0}", FormatScopedDiagnosticHandler::formatWithSource(diag)));
}

ErrorOrSuccess M::MOGG::verifyCustomOp(const MojoRegistry &registry,
                                       mlir::Operation *op) {
  MojoOperationInfo info = getMojoOperationInfo(op);
  if (info.operationName.empty())
    return Error(llvm::formatv("operation '{}' has no registered Mojo kernel",
                               op->getName().getStringRef())
                     .str());

  if (!registry.containsKernel(info.operationName)) {
    std::string message =
        llvm::formatv("no kernel registered for '{}'", info.operationName);
    if (auto suggestion = registry.suggestKernelName(info.operationName))
      message += llvm::formatv("; did you mean '{}'?", *suggestion).str();
    return Error(message);
  }

  // Catch MO-vs-kernel operand type mismatches before lowering emits Mojo.
  DeviceRefAttr device = MojoRegistry::getDeviceRefForOp(op);
  auto declOpOr = registry.getResolvedKernel(info.operationName, device);
  if (declOpOr.isError())
    return declOpOr.takeError();
  KernelDeclOp declOp = *declOpOr;
  Location structLoc = declOp.getMojoStructLocation();
  KernelDeclAdaptor adaptor(declOp);
  if (adaptor.isReduction()) {
    StringRef name = info.operationName;
    if (op->getNumOperands() != 1)
      return kernelError(structLoc,
                         llvm::formatv("reduction '{}' takes one input, got {}",
                                       name, op->getNumOperands()));
    size_t numFields = adaptor.reduction->getOutputs().size();
    if (op->getNumResults() != numFields)
      return kernelError(structLoc,
                         llvm::formatv("reduction '{}' has {} output fields, "
                                       "but the op has {} results",
                                       name, numFields, op->getNumResults()));
    ErrorOr<int64_t> axis =
        MO::getReductionAxis(op->getOperand(0), op->getResultTypes());
    if (axis.isError())
      return kernelError(
          structLoc, llvm::formatv("reduction '{}' {}", name, axis.getError()));
  } else {
    if (!adaptor.executeFunction)
      return kernelError(structLoc,
                         llvm::formatv("kernel '{}' has no execute function",
                                       info.operationName));
    Location executeLoc =
        declOp.getMojoFunctionLocation(KernelFunctionKind::Execute);
    KernelArgBlame blame;
    auto mappingOr =
        mapMOOpToKernelImplementation(op, *adaptor.executeFunction, blame);
    if (mappingOr.isError()) {
      Location loc =
          blame ? declOp.getMojoArgLocation(KernelFunctionKind::Execute, *blame)
                : executeLoc;
      return kernelError(loc, mappingOr.getError(), executeLoc,
                         llvm::formatv("in the 'execute' entry point of '{}'",
                                       info.operationName));
    }
  }

  // For mo.custom, additionally check that user-supplied params match the
  // kernel's compile-time parameter list (struct + fn-level).
  if (auto custom = mlir::dyn_cast<MO::CustomOp>(op)) {
    ErrorOrSuccess params = validateCustomOpParameters(registry, custom);
    if (params.isError())
      return kernelError(structLoc, params.getError());
  }

  return success();
}

ErrorOrSuccess M::MOGG::verifyCustomOp(mlir::ModuleOp kernelDecls,
                                       mlir::Operation *op) {
  MojoRegistry registry(kernelDecls);
  return verifyCustomOp(registry, op);
}
