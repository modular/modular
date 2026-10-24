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
//
// MOGG kernel signature introspection: `KernelDeclAdaptor` decodes the
// `FnTypeGeneratorType` signatures stored on a `mogg.kernel.decl`.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H

#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <cassert>
#include <optional>
#include <type_traits>
#include <variant>

namespace M::MOGG {

enum class IOSpec {
  InputTensor,  // Read-only
  OutputTensor, // Write-only
  MutableInputTensor,
  FusedInputTensor,
  FusedOutputTensor,
  FusedComputeOutputTensor,
  FusedComputeOutputTileTensor,
  FusedOutputTileTensor,
};

//===----------------------------------------------------------------------===//
// Tensor-argument trait contract (trait lane)
//
// Classifies a trait-bound tensor argument from the trait composition written
// on its `!lit.trait<...>` type, independently of the `IOSpec` decode.
//===----------------------------------------------------------------------===//

/// Role a tensor argument plays in the graph signature. `none` is required for
/// shape-function arguments, which read only metadata; an `execute` argument
/// requires exactly one of the other roles.
enum class TensorArgRole { input, output, mutableInput, none };

/// Element-access form a tensor argument exposes (`none` for a plain dense
/// argument). Also indexes the bitmasks of `TensorArgContract`.
enum class TensorArgAccess { none, load, store, transform };

/// The deepest trait bound written on a shape-function tensor argument:
/// `Tensor` (shape only) or `TileTensorable` (adds layout and host-side data
/// access). `unspecified` in the execute lane.
enum class TensorArgBound : uint8_t { unspecified, tensor, tile };

inline unsigned bit(TensorArgAccess access) {
  switch (access) {
  case TensorArgAccess::none:
    return 0;
  case TensorArgAccess::load:
    return 1u << 0;
  case TensorArgAccess::store:
    return 1u << 1;
  case TensorArgAccess::transform:
    return 1u << 2;
  }
  return 0;
}

/// The distilled contract for a trait-bound kernel argument. A non-fused dense
/// argument may carry several access forms (e.g. `load` and `store` for
/// read-modify-write); a fused argument exactly one.
struct TensorArgContract {
  TensorArgRole role = TensorArgRole::input;
  bool fused = false;
  bool stable = false;
  unsigned accessMask = 0;
  /// Subset of `accessMask` whose capability (e.g. `Loadable`) was required,
  /// not just the marker (e.g. `LoadAccess`). Matters for opaque-device
  /// targets, where only the marker may be available host-side.
  unsigned capabilityMask = 0;
  TensorArgBound bound = TensorArgBound::unspecified;

  bool isMutable() const {
    return role == TensorArgRole::output || role == TensorArgRole::mutableInput;
  }

  bool hasAccess(TensorArgAccess access) const {
    return (accessMask & bit(access)) != 0;
  }
  bool hasCapability(TensorArgAccess access) const {
    return (capabilityMask & bit(access)) != 0;
  }
  bool hasAnyAccess() const { return accessMask != 0; }
  int accessCount() const { return llvm::popcount(accessMask); }
  /// Requires `accessCount() == 1`, which a fused arg guarantees.
  TensorArgAccess soleAccess() const {
    assert(accessCount() == 1 && "soleAccess requires exactly one access form");
    for (TensorArgAccess a : {TensorArgAccess::load, TensorArgAccess::store,
                              TensorArgAccess::transform})
      if (hasAccess(a))
        return a;
    return TensorArgAccess::none;
  }

  bool operator==(const TensorArgContract &o) const {
    return role == o.role && fused == o.fused && stable == o.stable &&
           accessMask == o.accessMask && capabilityMask == o.capabilityMask &&
           bound == o.bound;
  }
  bool operator!=(const TensorArgContract &o) const { return !(*this == o); }
};

/// Distills the written trait composition of a trait-bound tensor argument
/// into a `TensorArgContract`.
///
/// `writtenSymbols` must come from `extensibility.tensor_arg_traits`; foreign
/// traits (e.g. `Copyable`) and unrecognized leaves are rejected via
/// `emitError`. The written composition and its ancestor-inclusive canonical
/// form distill identically; `AnyType` is ignored.
///
/// With `isShapeFunc`, the argument distills to `TensorArgRole::none` and may
/// bind only `Tensor` / `TileTensorable` (shape functions run
/// on the host, so data access is safe), recorded in `bound`. io-role markers,
/// `Fused`, and the fusion access traits are errors. Otherwise exactly one
/// io-role is required.
mlir::FailureOr<TensorArgContract> distillTensorArgContract(
    llvm::ArrayRef<M::KGEN::TraitSymbolAttr> writtenSymbols,
    llvm::function_ref<void(const llvm::Twine &)> emitError,
    bool isShapeFunc = false);

/// Returns the `!lit.trait<...>` bound if `argType` is a `!lit.ref` over a
/// trait-bound generic type parameter, else null.
KGEN::LIT::TraitType getTensorArgTraitType(mlir::Type argType);

/// If `argType` is a variadic pack of trait-bound tensor arguments -- the
/// `VariadicPack` behind `*xs: *Ts` with `*Ts` bound by tensor-argument traits
/// -- return the `TraitType` every element is bound by. Otherwise return null.
/// Not a supported spelling: import uses this to reject it in favor of
/// `TensorTuple`.
KGEN::LIT::TraitType getTensorArgPackTraitType(mlir::Type argType);

/// If `argType` is a `TensorTuple` of trait-bound tensor arguments --
/// `xs: TensorTuple[*Ts]` with `Ts: TypeList[Trait=..., ...]` bound by
/// tensor-argument traits -- return the `TraitType` every element is bound by.
/// Otherwise return null. The tuple distills to one contract shared by its
/// elements.
KGEN::LIT::TraitType getTensorArgTensorTupleTraitType(mlir::Type argType);

/// The trait symbols a trait-bound tensor argument distills from, in either
/// spelling: a single argument or a `TensorTuple`. Empty if `argType` is not a
/// trait-bound tensor argument.
llvm::SmallVector<M::KGEN::TraitSymbolAttr>
getTensorArgTraitSymbols(mlir::Type argType);

llvm::StringRef toString(TensorArgRole role);
llvm::StringRef toString(TensorArgAccess access);
llvm::StringRef toString(TensorArgBound bound);

/// Encodes the per-argument dict baked into
/// `KernelFunctionAttr.tensor_arg_contracts`. `bound` is omitted when
/// `unspecified`.
mlir::DictionaryAttr tensorArgContractToDict(mlir::MLIRContext *ctx,
                                             const TensorArgContract &contract);

/// Builds the attribute carried by `mogg.tensor_arg.__init__`.
MOGG::TensorArgContractAttr
tensorArgContractToAttr(mlir::MLIRContext *ctx,
                        const TensorArgContract &contract);

/// Inverse of `tensorArgContractToDict`. Returns `std::nullopt` for the
/// `UnitAttr` non-trait placeholder or any unexpected attr. A missing `bound`
/// decodes as `unspecified`.
std::optional<TensorArgContract> dictToTensorArgContract(mlir::Attribute attr);

//===----------------------------------------------------------------------===//
// Type predicates and IOSpec helpers
//===----------------------------------------------------------------------===//

/// Returns the underlying legacy tensor struct, including fused and stable
/// aliases, or null for other argument types. Does not require bound IO params.
KGEN::LIT::StructType getLegacyTensorArgType(mlir::Type argType);

/// Check whether a symbol has the given root and leaf, ignoring intermediate
/// refs, for symbols whose package depth varies (e.g. DeviceContext under
/// max::gpu::host may be re-exported more shallowly).
inline bool symbolMatchesRootAndLeaf(mlir::SymbolRefAttr symbol,
                                     llvm::StringRef root,
                                     llvm::StringRef leaf) {
  return symbol.getRootReference() == root && symbol.getLeafReference() == leaf;
}

inline bool isXType(KGEN::LIT::StructType maybeTensor, llvm::StringLiteral root,
                    llvm::StringLiteral className) {
  return symbolMatchesRootAndLeaf(maybeTensor.getSymbol(), root, className);
}

inline bool isExtensibilityTensor(KGEN::LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "extensibility", "Tensor");
}

inline bool isCustomType(KGEN::LIT::StructType maybeCustom) {
  return !isExtensibilityTensor(maybeCustom);
}

llvm::StringRef toString(IOSpec spec);
std::optional<IOSpec> toIOSpec(llvm::StringRef str);
std::pair<mlir::TypedAttr, mlir::TypedAttr> getParams(IOSpec ioSpec,
                                                      mlir::Builder &builder);

bool isOutputIOSpec(IOSpec spec);
bool isFusableIOSpec(IOSpec spec);
bool isMutableIOSpec(IOSpec spec);

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor: reads mogg.kernel.decl op signatures directly.
//===----------------------------------------------------------------------===//

struct KernelDeclTensorAdaptor {
  bool mut = false;
  bool fused = false;
  bool stable = false;
  IOSpec ioSpec = {};
  /// Set only for a trait-bound argument, which lowers to
  /// `mogg.tensor_arg.__init__`. The fields above hold equivalent values so
  /// the predicates behave the same for both kinds.
  std::optional<TensorArgContract> traitContract;
  KernelDeclTensorAdaptor() = default;
  explicit KernelDeclTensorAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}
  bool operator==(const KernelDeclTensorAdaptor &o) const {
    return mut == o.mut;
  }
};

struct KernelDeclVariadicTensorAdaptor {
  bool mut = false;
  bool fused = false;
  bool stable = false;
  IOSpec ioSpec = {};
  /// Present only for a trait-lane variadic (`xs: TensorTuple[*Ts]`): the
  /// contract shared by every element. Empty for the legacy `VariadicTensors`
  /// lane.
  std::optional<TensorArgContract> traitContract;
  KernelDeclVariadicTensorAdaptor() = default;
  explicit KernelDeclVariadicTensorAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}
  bool operator==(const KernelDeclVariadicTensorAdaptor &o) const {
    return mut == o.mut;
  }
};

struct KernelDeclScalarAdaptor {
  bool operator==(const KernelDeclScalarAdaptor &) const { return true; }
};

struct KernelDeclTileTensorAdaptor {
  bool operator==(const KernelDeclTileTensorAdaptor &) const { return true; }
};

struct KernelDeclContextPtrAdaptor {
  bool operator==(const KernelDeclContextPtrAdaptor &) const { return true; }
};

struct KernelDeclContextPtrListAdaptor {
  bool operator==(const KernelDeclContextPtrListAdaptor &) const {
    return true;
  }
};

struct KernelDeclGraphBuilderAdaptor {
  bool operator==(const KernelDeclGraphBuilderAdaptor &) const { return true; }
};

struct KernelDeclUnsupportedAdaptor {
  bool operator==(const KernelDeclUnsupportedAdaptor &) const { return false; }
};

/// Opaque argument type; its name is owned by the MLIR context via StringAttr.
struct KernelDeclOpaqueAdaptor {
  mlir::StringAttr typeName;
  bool operator==(const KernelDeclOpaqueAdaptor &o) const {
    if (typeName == o.typeName)
      return true;
    llvm::StringRef a = typeName.strref(), b = o.typeName.strref();
    return a.ends_with(b) || b.ends_with(a);
  }
};

using KernelDeclOperandVariant =
    std::variant<KernelDeclTensorAdaptor, KernelDeclVariadicTensorAdaptor,
                 KernelDeclScalarAdaptor, KernelDeclTileTensorAdaptor,
                 KernelDeclOpaqueAdaptor, KernelDeclContextPtrAdaptor,
                 KernelDeclContextPtrListAdaptor, KernelDeclGraphBuilderAdaptor,
                 KernelDeclUnsupportedAdaptor>;

/// One argument decoded from a kernel's stored LIT signature.
struct KernelDeclOperandAdaptor {
  struct SourceDescriptor {
    mlir::StringAttr sourceName;
    uint64_t position;
    bool isByRefResult;
  };

  std::optional<SourceDescriptor> sourceDescriptor;
  KernelDeclOperandVariant underlyingType;

  bool isDpsOutput() const {
    return sourceDescriptor.has_value() && !sourceDescriptor->isByRefResult;
  }

  bool isTensorType() const {
    return std::holds_alternative<KernelDeclTensorAdaptor>(underlyingType) ||
           std::holds_alternative<KernelDeclVariadicTensorAdaptor>(
               underlyingType);
  }

  bool isVariadicTensorType() const {
    return std::holds_alternative<KernelDeclVariadicTensorAdaptor>(
        underlyingType);
  }

  bool isMutableTensorType() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->mut;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->mut;
    return false;
  }

  bool isStableTensorType() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->stable;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->stable;
    return false;
  }

  bool isFusedTensorType() const {
    // The contract is the source of truth for trait-bound args. Prologue vs
    // epilogue follows from operand position, and the access form is resolved
    // at emit.
    if (std::optional<TensorArgContract> contract = getTraitContract())
      return contract->fused;
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->fused;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->fused;
    return false;
  }

  /// nullopt for a `ManagedTensorSlice` or non-tensor argument.
  std::optional<TensorArgContract> getTraitContract() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->traitContract;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->traitContract;
    return std::nullopt;
  }

  bool isTraitTensorType() const { return getTraitContract().has_value(); }

  /// nullopt for a non-tensor argument.
  std::optional<IOSpec> getIoSpec() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->ioSpec;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->ioSpec;
    return std::nullopt;
  }

  /// A trait-lane `TensorTuple` (`xs: TensorTuple[*Ts]`), whose elements are
  /// gathered into one argument.
  bool isTraitTensorTupleType() const {
    return isVariadicTensorType() && isTraitTensorType();
  }

  bool isScalarType() const {
    return std::holds_alternative<KernelDeclScalarAdaptor>(underlyingType);
  }

  bool isTileType() const {
    return std::holds_alternative<KernelDeclTileTensorAdaptor>(underlyingType);
  }

  bool isOpaqueType() const {
    return std::holds_alternative<KernelDeclOpaqueAdaptor>(underlyingType);
  }

  bool isContextType() const {
    return std::holds_alternative<KernelDeclContextPtrAdaptor>(
               underlyingType) ||
           std::holds_alternative<KernelDeclContextPtrListAdaptor>(
               underlyingType);
  }

  bool isVariadicContextType() const {
    return std::holds_alternative<KernelDeclContextPtrListAdaptor>(
        underlyingType);
  }

  bool isBuilderType() const {
    return std::holds_alternative<KernelDeclGraphBuilderAdaptor>(
        underlyingType);
  }

  /// The trailing "where to run" argument (a DeviceContext[Array], or a
  /// DeviceGraphBuilder for `build_graph`). Must appear last and is skipped
  /// when aligning inputs across function variants.
  bool isLaunchArgument() const { return isContextType() || isBuilderType(); }

  bool isUnsupportedType() const {
    return std::holds_alternative<KernelDeclUnsupportedAdaptor>(underlyingType);
  }

  bool isOpaqueWithTypeName(llvm::StringRef name) const {
    const auto *o = std::get_if<KernelDeclOpaqueAdaptor>(&underlyingType);
    if (!o)
      return false;
    // Strict equality: `name` must use buildOpaqueSymbol's
    // "root[.intermediate...]::leaf" form (e.g.
    // kMojoInternalDpsIndexListTypeQualifiedName).
    return o->typeName.strref() == name;
  }

  bool operator==(const KernelDeclOperandAdaptor &other) const {
    return underlyingType == other.underlyingType;
  }

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    if (sourceDescriptor.has_value())
      os << nesting << sourceDescriptor->sourceName.strref().str() << ": ";
    os << underlyingType;
    return os;
  }
};

/// Decoded view of one function (execute / shape / elementwise), built from
/// its FnTypeGeneratorType alone. `attr` is null for synthesized adaptors
/// (see `synthesizeExecuteFromElementwise`), so callers that may see one must
/// null-check.
struct KernelDeclFunctionAdaptor {
  KernelFunctionAttr attr;
  llvm::SmallVector<KernelDeclOperandAdaptor> inputArguments;
  llvm::SmallVector<KernelDeclOperandAdaptor> outputArguments;
  std::optional<KernelDeclOperandAdaptor> outputResult;
  /// e.g. `max.kernels.AddOp.execute`. Null for synthesized adaptors.
  mlir::StringAttr fullyResolvedName;

  /// `tensorArgContracts` is the import-baked
  /// `KernelFunctionAttr.tensor_arg_contracts`; pass it when available. When
  /// empty (primitives, synthesized adaptors), the written composition is
  /// re-distilled.
  explicit KernelDeclFunctionAdaptor(KGEN::LIT::FnTypeGeneratorType sig,
                                     mlir::ArrayAttr tensorArgContracts = {});

  KernelDeclFunctionAdaptor(KernelFunctionAttr attr,
                            StringAttr fullyResolvedName);

  KernelDeclFunctionAdaptor(ShapeFunctionDeclOp shapeDecl);

  /// Build an execute-compatible adaptor from an elementwise signature:
  /// promotes SIMD arguments to fused tensors, strips a trailing IndexList
  /// argument, and injects a synthetic fused-output tensor.
  static KernelDeclFunctionAdaptor
  synthesizeExecuteFromElementwise(KGEN::LIT::FnTypeGeneratorType sig);

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    os << nesting << "(\n";
    for (const auto &arg : outputArguments)
      arg.printNested(os, nesting + "  [out] ") << ",\n";
    for (const auto &arg : inputArguments)
      arg.printNested(os, nesting + "  [in]  ") << ",\n";
    os << nesting << ")";
    if (outputResult.has_value())
      os << " -> " << *outputResult;
    return os;
  }

private:
  KernelDeclFunctionAdaptor() = default;
};

/// Decodes every function signature of a mogg.kernel.decl from its stored LIT
/// types.
struct KernelDeclAdaptor {
  mlir::StringAttr registeredOpName;
  /// e.g. `max.kernels.AddOp`.
  mlir::StringAttr fullyResolvedName;
  std::optional<KernelDeclFunctionAdaptor> executeFunction;
  std::optional<KernelDeclFunctionAdaptor> elementwiseFunction;
  std::optional<KernelDeclFunctionAdaptor> tileElementwiseFunction;
  std::optional<KernelDeclFunctionAdaptor> buildGraphFunction;
  std::optional<KernelDeclFunctionAdaptor> executeDeviceFunction;
  std::optional<KernelDeclFunctionAdaptor> executeConfigurationFunction;
  /// Set for a `ReduceOp` monoid.
  std::optional<KernelReductionAttr> reduction;
  /// Struct-level ParamDeclAttrs shared by all functions (e.g. `dtype`).
  mlir::ArrayAttr structParams;
  /// The kernel also declares an `execute` overload taking its fused outputs
  /// as plain outputs, for a call with no epilogue fusion (see
  /// `KernelDeclOp::supports_unfused_execute`). Not a stored signature:
  /// `ConformCallOperands` derives it from `executeFunction`.
  bool supportsUnfusedExecute = false;
  /// Whether calls launch `execute_device` rather than call `execute`.
  /// `executeFunction` is then `execute_device`'s signature, so callers that
  /// map operands through it work for either entry point.
  bool isDeclarative() const { return executeDeviceFunction.has_value(); }

  explicit KernelDeclAdaptor(KernelDeclOp decl);

  bool isElementwise() const { return elementwiseFunction.has_value(); }
  bool isTileElementwise() const { return tileElementwiseFunction.has_value(); }
  bool isReduction() const { return reduction.has_value(); }

  /// Returns the kernel's `kind` function, or nullptr if it has none. A shape
  /// function lives in its own decl, so `Shape` is always nullptr here.
  const KernelDeclFunctionAdaptor *getFunction(KernelFunctionKind kind) const;
};

/// Adaptor for a mogg.primitive.decl: a standalone @register_internal Mojo
/// function (e.g. the `foreach` driver).
struct KernelFunctionDeclAdaptor {
  mlir::StringAttr registration;
  KernelDeclFunctionAdaptor function;

  explicit KernelFunctionDeclAdaptor(PrimitiveDeclOp decl);
};

//===----------------------------------------------------------------------===//
// Printers
//===----------------------------------------------------------------------===//

/// Uses if-constexpr to keep ADL confined to M::MOGG.
template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclOperandVariant &var) {
  std::visit(
      [&os](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, KernelDeclTensorAdaptor>) {
          os << "Tensor[mut=" << (v.mut ? "true" : "false")
             << ", fused=" << (v.fused ? "true" : "false")
             << (v.stable ? ", stable" : "") << "]";
        } else if constexpr (std::is_same_v<T,
                                            KernelDeclVariadicTensorAdaptor>) {
          os << "*Tensor[mut=" << (v.mut ? "true" : "false")
             << ", fused=" << (v.fused ? "true" : "false")
             << (v.stable ? ", stable" : "") << "]";
        } else if constexpr (std::is_same_v<T, KernelDeclScalarAdaptor>) {
          os << "Scalar";
        } else if constexpr (std::is_same_v<T, KernelDeclOpaqueAdaptor>) {
          os << "<" << v.typeName.strref() << ">";
        } else if constexpr (std::is_same_v<T, KernelDeclContextPtrAdaptor>) {
          os << "Context";
        } else if constexpr (std::is_same_v<T,
                                            KernelDeclContextPtrListAdaptor>) {
          os << "*Context";
        } else if constexpr (std::is_same_v<T, KernelDeclGraphBuilderAdaptor>) {
          os << "GraphBuilder";
        } else if constexpr (std::is_same_v<T, KernelDeclUnsupportedAdaptor>) {
          os << "Unsupported";
        }
      },
      var);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclOperandAdaptor &op) {
  return op.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclFunctionAdaptor &fn) {
  return fn.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclAdaptor &adaptor) {
  os << "@register(" << adaptor.registeredOpName.strref() << ")\n";
  if (adaptor.executeFunction.has_value()) {
    os << "execute";
    adaptor.executeFunction->printNested(os, "  ");
    os << "\n";
  }
  if (adaptor.isReduction()) {
    os << "reduction " << adaptor.registeredOpName.strref()
       << ": width = " << adaptor.reduction->getWidthParam().strref()
       << ", outputs = [";
    llvm::StringRef sep = "";
    for (mlir::Attribute output : adaptor.reduction->getOutputs()) {
      os << sep
         << mlir::cast<KernelReductionOutputAttr>(output).getName().strref();
      sep = ", ";
    }
    os << "]\n";
  }
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const KernelFunctionDeclAdaptor &adaptor) {
  os << "@primitive(" << adaptor.registration.strref() << ")\n";
  os << "function";
  adaptor.function.printNested(os, "  ");
  os << "\n";
  return os;
}

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H
