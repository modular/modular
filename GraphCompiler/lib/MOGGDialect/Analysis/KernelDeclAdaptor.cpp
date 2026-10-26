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

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <iterator>

using namespace M;
using namespace M::MOGG;
using namespace M::KGEN;

namespace {

// Symbol path arrays for recognising known struct types. Components live in
// MojoIdentifiers.h (`kPackage*`, `kLeaf*`).
static constexpr std::array<llvm::StringLiteral, 3> kMaxManagedTensorSlice = {
    kPackageExtensibility, kPackageManagedTensorSlice, kLeafManagedTensorSlice};
static constexpr std::array<llvm::StringLiteral, 3> kMaxVariadicTensors = {
    kPackageExtensibility, kPackageManagedTensorSlice, kLeafVariadicTensors};
static constexpr std::array<llvm::StringLiteral, 3>
    kMaxFusedInputVariadicTensors = {kPackageExtensibility,
                                     kPackageManagedTensorSlice,
                                     kLeafFusedInputVariadicTensors};
static constexpr std::array<llvm::StringLiteral, 3>
    kMaxFusedOutputVariadicTensors = {kPackageExtensibility,
                                      kPackageManagedTensorSlice,
                                      kLeafFusedOutputVariadicTensors};
static constexpr std::array<llvm::StringLiteral, 3> kMaxStableTensor = {
    kPackageExtensibility, kPackageStableTensor, kLeafStableTensor};
static constexpr std::array<llvm::StringLiteral, 3> kMaxStableVariadicTensors =
    {kPackageExtensibility, kPackageStableTensor, kLeafStableVariadicTensors};

static constexpr llvm::StringLiteral kLeafDeviceContextBase =
    "DeviceContextBase";

// Dict keys shared by `tensorArgContractToDict` (writer) and
// `dictToTensorArgContract` (reader).
static constexpr llvm::StringLiteral kRoleKey = "role";
static constexpr llvm::StringLiteral kFusedKey = "fused";
static constexpr llvm::StringLiteral kAccessKey = "access";
static constexpr llvm::StringLiteral kCapabilitiesKey = "capabilities";
static constexpr llvm::StringLiteral kBoundKey = "bound";
static constexpr llvm::StringLiteral kStableKey = "stable";

/// Build a "root.intermediate...leaf" opaque type string from a symbol ref,
/// '.'-joined like a Mojo qualified name, so consumers (e.g.
/// mlirOpaqueTypeToMojoType) use it directly.
static mlir::StringAttr buildOpaqueSymbol(mlir::MLIRContext *ctx,
                                          mlir::SymbolRefAttr sym) {
  std::string result = sym.getRootReference().strref().str();
  for (auto ref : sym.getNestedReferences())
    result += "." + ref.getValue().str();
  return mlir::StringAttr::get(ctx, result);
}

/// Check whether a symbol matches a dotted-path array (root + nested refs).
static bool symbolMatches(mlir::SymbolRefAttr symbol,
                          llvm::ArrayRef<llvm::StringLiteral> path) {
  if (symbol.getNestedReferences().size() != path.size() - 1)
    return false;
  if (!symbol.getRootReference().strref().contains(path.front()))
    return false;
  for (auto [i, ref] : llvm::enumerate(symbol.getNestedReferences())) {
    if (!ref.getValue().contains(path[i + 1]))
      return false;
  }
  return true;
}

/// Strip LIT::RefType sugar and return the underlying StructType, or null.
static KGEN::LIT::StructType getAsStructType(mlir::Type t) {
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(t))
    return KGEN::sugarDynCast<KGEN::LIT::StructType>(ref.getElementType());
  return KGEN::sugarDynCast<KGEN::LIT::StructType>(t);
}

/// Extract the IOSpec encoded in the mut/input parameter values of a
/// ManagedTensorSlice or VariadicTensors struct type.  Returns std::nullopt
/// when the parameters are not yet bound.
static std::optional<IOSpec>
getIOSpecFromStructType(KGEN::LIT::StructType structType) {
  // Fused variadic types carry their IOSpec in the symbol name alone.
  if (symbolMatches(structType.getSymbol(), kMaxFusedInputVariadicTensors))
    return IOSpec::FusedInputTensor;
  if (symbolMatches(structType.getSymbol(), kMaxFusedOutputVariadicTensors))
    return IOSpec::FusedOutputTensor;

  // All other tensors (including the slot-backed `StableTensor` /
  // `StableVariadicTensors`) encode mut and input as the first two param
  // values.
  auto sym = structType.getSymbol();
  bool isTensorStruct = symbolMatches(sym, kMaxManagedTensorSlice) ||
                        symbolMatches(sym, kMaxVariadicTensors) ||
                        symbolMatches(sym, kMaxStableTensor) ||
                        symbolMatches(sym, kMaxStableVariadicTensors);
  if (!isTensorStruct)
    return std::nullopt;

  auto params = structType.getParamValues();
  if (params.size() < 2)
    return std::nullopt;

  mlir::TypedAttr mutAttr =
      KGEN::getCanonicalAttr(mlir::cast<mlir::TypedAttr>(params[0]));
  mlir::TypedAttr rawInputAttr = mlir::cast<mlir::TypedAttr>(params[1]);

  // mut is a Bool struct: { value: scalar<bool> }.
  auto mutStruct = KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(mutAttr);
  if (!mutStruct || mutStruct.getValues().empty())
    return std::nullopt;
  auto [_1, mutValueAttr] = mutStruct.getValues().front();
  auto mutBoolAttr = KGEN::sugarDynCast<KGEN::SIMDAttr>(mutValueAttr);
  if (!mutBoolAttr)
    return std::nullopt;
  bool mut = mutBoolAttr.getAsBool();

  int64_t input = kIOSpecIOUnknown;

  // In the primary module (no dep elaboration) the IO param is a
  // SugarAttr(MemberAlias) whose member name encodes the IO variant. Check it
  // before canonicalisation, which yields a ParamOperatorAttr like IO(3).
  if (auto ioSugar = mlir::dyn_cast<KGEN::SugarAttr>(rawInputAttr)) {
    if (ioSugar.getKind() == KGEN::SugarKind::MemberAlias) {
      llvm::StringRef name = ioSugar.getMemberName().strref();
      if (name == kIoOutput)
        input = kIOSpecIOOutput;
      else if (name == kIoInput)
        input = kIOSpecIOInput;
      else if (name == kIoFusedInput)
        input = kIOSpecIOFusedInput;
      else if (name == kIoFusedOutput)
        input = kIOSpecIOFusedOutput;
      else if (name == kIoFusedComputeOutput)
        input = kIOSpecIOFusedComputeOutput;
      else if (name == kIoFusedComputeOutputTile)
        input = kIOSpecIOFusedComputeOutputTile;
      else if (name == kIoFusedOutputTile)
        input = kIOSpecIOFusedOutputTile;
    }
  }

  // Fallback path: in elaborated bytecode the IO param is a fully-resolved
  // LITStructAttr { value: SIMDLength { value: i64 } } (two levels of nesting).
  if (input == kIOSpecIOUnknown) {
    mlir::TypedAttr inputAttr = KGEN::getCanonicalAttr(rawInputAttr);
    auto inputStruct = KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(inputAttr);
    if (inputStruct && !inputStruct.getValues().empty()) {
      auto [_2, inputValueAttr] = inputStruct.getValues().front();
      auto innerStruct =
          KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(inputValueAttr);
      if (innerStruct && !innerStruct.getValues().empty()) {
        auto [_3, inputIntValueAttr] = innerStruct.getValues().front();
        auto inputIntAttr =
            KGEN::sugarDynCast<mlir::IntegerAttr>(inputIntValueAttr);
        if (inputIntAttr)
          input = inputIntAttr.getValue().getSExtValue();
      }
    }
  }

  if (input == kIOSpecIOUnknown)
    return std::nullopt;

  if (mut == kIOSpecMutable && input == kIOSpecIOOutput)
    return IOSpec::OutputTensor;
  if (mut == kIOSpecImmutable && input == kIOSpecIOInput)
    return IOSpec::InputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOInput)
    return IOSpec::MutableInputTensor;
  if (mut == kIOSpecImmutable && input == kIOSpecIOFusedInput)
    return IOSpec::FusedInputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedOutput)
    return IOSpec::FusedOutputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedComputeOutput)
    return IOSpec::FusedComputeOutputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedComputeOutputTile)
    return IOSpec::FusedComputeOutputTileTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedOutputTile)
    return IOSpec::FusedOutputTileTensor;
  return std::nullopt;
}

/// If `argType` is a trait-bound tensor argument, re-derive its
/// `TensorArgContract` from the written trait composition, as
/// `distillTensorArgContract` does at import (errors are swallowed: import
/// already reported them). Returns nullopt for a non-trait arg or an
/// unresolvable composition.
///
/// Fallback for decls without an import-baked `tensor_arg_contracts` array
/// (primitives, synthesized adaptors); see `resolveTraitContract`.
static std::optional<TensorArgContract>
recoverTraitArgContract(mlir::Type argType) {
  llvm::SmallVector<TraitSymbolAttr> symbols =
      getTensorArgTraitSymbols(argType);
  if (symbols.empty())
    return std::nullopt;
  mlir::FailureOr<TensorArgContract> contract =
      distillTensorArgContract(symbols, [](const llvm::Twine &) {});
  if (mlir::failed(contract))
    return std::nullopt;
  return *contract;
}

/// Build a `KernelDeclTensorAdaptor` with the role/mut implied by a trait arg's
/// contract, so isTensorType/isMutableTensorType/isDpsOutput behave as for the
/// equivalent `IOSpec`: `output -> OutputTensor`, `mutable_input ->
/// MutableInputTensor`, `input -> InputTensor`. Fusion isn't modeled
/// (dense-only); `fused` stays false.
static KernelDeclTensorAdaptor
traitContractToTensorAdaptor(const TensorArgContract &contract) {
  KernelDeclTensorAdaptor adaptor;
  switch (contract.role) {
  case TensorArgRole::output:
    adaptor = KernelDeclTensorAdaptor{IOSpec::OutputTensor};
    break;
  case TensorArgRole::mutableInput:
    adaptor = KernelDeclTensorAdaptor{IOSpec::MutableInputTensor};
    break;
  case TensorArgRole::input:
  case TensorArgRole::none:
    // The no-role shape-function flavor is read-only, so it maps to
    // `InputTensor`; the contract attached below keeps `role == none`.
    adaptor = KernelDeclTensorAdaptor{IOSpec::InputTensor};
    break;
  }
  // Lets lowering route this arg to `mogg.tensor_arg.__init__`, unlike a
  // `ManagedTensorSlice` arg with the same `ioSpec`.
  adaptor.traitContract = contract;
  return adaptor;
}

/// The variadic counterpart of `traitContractToTensorAdaptor`, for a
/// `TensorTuple`.
static KernelDeclVariadicTensorAdaptor
traitContractToVariadicAdaptor(const TensorArgContract &contract) {
  KernelDeclTensorAdaptor scalar = traitContractToTensorAdaptor(contract);
  KernelDeclVariadicTensorAdaptor adaptor{scalar.ioSpec};
  adaptor.traitContract = contract;
  return adaptor;
}

/// Resolve a trait-bound tensor argument's contract. The import-baked
/// `KernelFunctionAttr.tensor_arg_contracts` dict is authoritative when present
/// (`hasBakedArray`): a `UnitAttr` placeholder decodes to nullopt, a dict to
/// its contract. Without one (primitives / synthesized adaptors), re-distill
/// the written composition. The two are never mixed for one function.
static std::optional<TensorArgContract>
resolveTraitContract(mlir::Type argType, mlir::Attribute bakedEntry,
                     bool hasBakedArray) {
  if (hasBakedArray)
    return dictToTensorArgContract(bakedEntry);
  return recoverTraitArgContract(argType);
}

/// Classify a single argument type into a KernelDeclOperandVariant by
/// introspecting the LIT StructType symbol and parameter values.
///
/// `bakedEntry` (this argument's baked entry, or null) and `hasBakedArray`
/// feed `resolveTraitContract`.
static KernelDeclOperandVariant classifyArgType(mlir::MLIRContext *ctx,
                                                mlir::Type argType,
                                                bool promoteSIMDToFusedTensor,
                                                mlir::Attribute bakedEntry,
                                                bool hasBakedArray) {
  KGEN::LIT::StructType structType = getAsStructType(argType);
  if (!structType) {
    // A trait-bound tensor argument is non-struct; classify it as a tensor
    // adaptor with its contract's role/mut so it joins operand mapping and the
    // DPS-output split. Other non-struct args are unsupported.
    if (std::optional<TensorArgContract> contract =
            resolveTraitContract(argType, bakedEntry, hasBakedArray))
      return traitContractToTensorAdaptor(*contract);
    return KernelDeclUnsupportedAdaptor{};
  }

  // Trait lane, variadic: `xs: TensorTuple[*Ts]` is a `TensorTuple` struct. It
  // carries one contract for all its elements.
  if (getTensorArgTensorTupleTraitType(argType)) {
    if (std::optional<TensorArgContract> contract =
            resolveTraitContract(argType, bakedEntry, hasBakedArray))
      return traitContractToVariadicAdaptor(*contract);
    return KernelDeclUnsupportedAdaptor{};
  }

  auto sym = structType.getSymbol();

  // --- Tensors ---
  if (symbolMatches(sym, kMaxManagedTensorSlice)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    return KernelDeclTensorAdaptor{*ioSpec};
  }

  if (symbolMatches(sym, kMaxVariadicTensors)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    return KernelDeclVariadicTensorAdaptor{*ioSpec};
  }

  if (symbolMatches(sym, kMaxFusedInputVariadicTensors))
    return KernelDeclVariadicTensorAdaptor{IOSpec::FusedInputTensor};

  if (symbolMatches(sym, kMaxFusedOutputVariadicTensors))
    return KernelDeclVariadicTensorAdaptor{IOSpec::FusedOutputTensor};

  // Slot-backed views: same adaptor kinds as their pointer-backed siblings,
  // flagged `stable` so emit and the verifier can tell them apart.
  if (symbolMatches(sym, kMaxStableTensor)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    KernelDeclTensorAdaptor adaptor{*ioSpec};
    adaptor.stable = true;
    return adaptor;
  }

  if (symbolMatches(sym, kMaxStableVariadicTensors)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    KernelDeclVariadicTensorAdaptor adaptor{*ioSpec};
    adaptor.stable = true;
    return adaptor;
  }

  // --- Scalar ---
  // SIMD may be re-exported at different depths (e.g. @std::@builtin::@SIMD vs
  // @std::@SIMD).
  if (symbolMatchesRootAndLeaf(sym, kPackageStd, kLeafSIMD)) {
    if (promoteSIMDToFusedTensor)
      return KernelDeclTensorAdaptor{IOSpec::FusedInputTensor};
    return KernelDeclScalarAdaptor{};
  }

  // --- TileTensor ---
  if (sym.getLeafReference().strref() == kLeafTileTensor)
    return KernelDeclTileTensorAdaptor{};

  // --- Device context ---
  // Use root+leaf matching because DeviceContext lives under max::gpu::host
  // (depth varies across package exports).
  // `DeviceContext` is an alias of the target-parameterized
  // `DeviceContextBase`, which is the name the struct type carries.
  if (symbolMatchesRootAndLeaf(sym, kPackageMax, kLeafDeviceContext) ||
      symbolMatchesRootAndLeaf(sym, kPackageMax, kLeafDeviceContextBase))
    return KernelDeclContextPtrAdaptor{};

  if (symbolMatchesRootAndLeaf(sym, kPackageMax, kLeafDeviceContextArray))
    return KernelDeclContextPtrListAdaptor{};

  // --- Device graph builder (launch arg of build_graph) ---
  if (symbolMatchesRootAndLeaf(sym, kPackageMax, kLeafDeviceGraphBuilder))
    return KernelDeclGraphBuilderAdaptor{};

  // --- Opaque: unrecognised struct ---
  return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
}

/// Count the number of leading DPS output arguments (output-IOSpec tensors
/// that come before any input or non-tensor argument).
///
/// `tensorArgContracts` is the function's baked array (signature-arg order),
/// or null/empty if none; it drives trait args' role via
/// `resolveTraitContract`.
static size_t countDPSOutputs(KGEN::LIT::FnTypeGeneratorType sig,
                              mlir::ArrayAttr tensorArgContracts) {
  bool hasBakedArray = tensorArgContracts && !tensorArgContracts.empty();
  size_t count = 0;
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult)
      break;
    mlir::Type argType = sig.getArgument(i);
    KGEN::LIT::StructType structType = getAsStructType(argType);
    if (!structType || getTensorArgTensorTupleTraitType(argType)) {
      // A trait-bound tensor arg is non-struct and a trait variadic is a
      // `TensorTuple`: an `output` contract role continues the count; any other
      // role, or a non-trait non-struct arg, stops it.
      mlir::Attribute bakedEntry =
          (hasBakedArray && i < tensorArgContracts.size())
              ? tensorArgContracts[i]
              : mlir::Attribute();
      std::optional<TensorArgContract> contract =
          resolveTraitContract(argType, bakedEntry, hasBakedArray);
      if (!contract || contract->role != TensorArgRole::output)
        break;
      ++count;
      continue;
    }
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec || !isOutputIOSpec(*ioSpec))
      break;
    ++count;
  }
  return count;
}

//===----------------------------------------------------------------------===//
// Trait-lane vocabulary classification
//===----------------------------------------------------------------------===//

// How a vocabulary trait participates in the distilled contract. Validation
// is per category, never per trait, so adding a trait to an existing category
// is one `kTensorArgVocab` row; only a new kind of trait grows this enum.
enum class TraitCategory {
  /// An io-role marker (`Input` / `Output` / `MutableInput`); exactly one is
  /// required. The row's `role` payload names the role it contributes.
  role,
  /// The `Fused` fusion marker.
  fused,
  /// A metadata-only base trait (`Tensor`); carries no contract meaning.
  metadata,
  /// A dense data-access capability (`TileTensorable`); carries no contract
  /// meaning on its own.
  dataCapability,
  /// A storage marker (`Stable`): the argument's pointer lives in a stable
  /// address slot. Carries no payload; contributes `contract.stable`.
  storage,
  /// An access-form marker (`LoadAccess` / `StoreAccess`), or the
  /// `_LegacyComputeOutput` slot. The row's `access` payload names the form it
  /// contributes. Unlike the markers, the slot's `transform` form doesn't
  /// promise its capability: the slot may be delivered as a store fusion.
  accessMarker,
  /// An access-form callable capability (`Loadable` / `Storable` /
  /// `_LegacyComputable`). The row's `access` payload names the form.
  accessCapability,
  /// Reserved for the deferred tile-transform fusion traits; always rejected.
  deferred,
};

// One row of the tensor-argument trait vocabulary: the trait's leaf name, its
// category, and the category's payload (the role a `role` row contributes, or
// the access form an `accessMarker` / `accessCapability` row contributes).
struct VocabEntry {
  StringLiteral leaf;
  TraitCategory category;
  /// The io-role, for `role` rows.
  TensorArgRole role = TensorArgRole::input;
  /// The access form, for `accessMarker` / `accessCapability` rows.
  TensorArgAccess access = TensorArgAccess::none;

private:
  // Deliberately not constexpr: reaching it while building the constexpr
  // `kTensorArgVocab` rows is a compile error naming the violated invariant
  // and the offending row. Unreachable at runtime.
  static void payloadCategoryNeedsAPayload() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }
  static void onlyRoleRowsCarryARole() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }
  static void onlyAccessRowsCarryAForm() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }

public:
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category)
      : leaf(leaf), category(category) {
    if (category == TraitCategory::role ||
        category == TraitCategory::accessMarker ||
        category == TraitCategory::accessCapability)
      payloadCategoryNeedsAPayload();
  }
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category,
                       TensorArgRole role)
      : leaf(leaf), category(category), role(role) {
    if (category != TraitCategory::role)
      onlyRoleRowsCarryARole();
  }
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category,
                       TensorArgAccess access)
      : leaf(leaf), category(category), access(access) {
    if (category != TraitCategory::accessMarker &&
        category != TraitCategory::accessCapability)
      onlyAccessRowsCarryAForm();
    if (access == TensorArgAccess::none)
      onlyAccessRowsCarryAForm();
  }
};

// The tensor-argument trait vocabulary of `extensibility::tensor_arg_traits`.
// Keep in sync with tensor_arg_traits.mojo, except `TileTransformAccess` /
// `TileTransformable`, reserved for the deferred tile-transform fusion traits
// and not yet in the module (so the `deferred` rejection is unreachable).
// Leaf constant for the `isPresentLeaf` lookups, so call sites can't drift
// from the row spelling.
constexpr StringLiteral kTileTensorableLeaf = "TileTensorable";

constexpr VocabEntry kTensorArgVocab[] = {
    {"Input", TraitCategory::role, TensorArgRole::input},
    {"Output", TraitCategory::role, TensorArgRole::output},
    {"MutableInput", TraitCategory::role, TensorArgRole::mutableInput},
    {"Fused", TraitCategory::fused},
    {"Tensor", TraitCategory::metadata},
    {"LoadAccess", TraitCategory::accessMarker, TensorArgAccess::load},
    {"Loadable", TraitCategory::accessCapability, TensorArgAccess::load},
    {"StoreAccess", TraitCategory::accessMarker, TensorArgAccess::store},
    {"Storable", TraitCategory::accessCapability, TensorArgAccess::store},
    {"_LegacyComputeOutput", TraitCategory::accessMarker,
     TensorArgAccess::transform},
    {"_LegacyComputable", TraitCategory::accessCapability,
     TensorArgAccess::transform},
    {kTileTensorableLeaf, TraitCategory::dataCapability},
    {"Stable", TraitCategory::storage},
    {"TileTransformAccess", TraitCategory::deferred},
    {"TileTransformable", TraitCategory::deferred},
};

// Whole-table invariants, checked at compile time. The per-row invariants
// (payload/category coherence) live in the VocabEntry constructors; these
// cover what no single row can see.
// `StringRef::operator==` is not constexpr here; compare manually.
constexpr bool sameLeaf(StringLiteral a, StringLiteral b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a.data()[i] != b.data()[i])
      return false;
  return true;
}

constexpr bool leavesAreUnique() {
  for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
    for (size_t j = i + 1; j < std::size(kTensorArgVocab); ++j)
      if (sameLeaf(kTensorArgVocab[i].leaf, kTensorArgVocab[j].leaf))
        return false;
  return true;
}

constexpr bool payloadsAreUnique() {
  for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
    for (size_t j = i + 1; j < std::size(kTensorArgVocab); ++j) {
      const VocabEntry &a = kTensorArgVocab[i];
      const VocabEntry &b = kTensorArgVocab[j];
      if (a.category != b.category)
        continue;
      if (a.category == TraitCategory::role && a.role == b.role)
        return false;
      if ((a.category == TraitCategory::accessMarker ||
           a.category == TraitCategory::accessCapability) &&
          a.access == b.access)
        return false;
    }
  return true;
}

constexpr bool accessFormsArePaired() {
  // The distiller keys marker and capability rows of one form to each other;
  // a form must carry both rows.
  for (const VocabEntry &a : kTensorArgVocab) {
    if (a.category != TraitCategory::accessMarker &&
        a.category != TraitCategory::accessCapability)
      continue;
    TraitCategory wanted = a.category == TraitCategory::accessMarker
                               ? TraitCategory::accessCapability
                               : TraitCategory::accessMarker;
    bool paired = false;
    for (const VocabEntry &b : kTensorArgVocab)
      paired |= b.category == wanted && b.access == a.access;
    if (!paired)
      return false;
  }
  return true;
}

static_assert(leavesAreUnique(), "duplicate leaf in kTensorArgVocab");
static_assert(payloadsAreUnique(),
              "two kTensorArgVocab rows claim the same (category, payload)");
static_assert(accessFormsArePaired(),
              "unpaired access marker/capability row in kTensorArgVocab");

// Classify a trait symbol as a tensor-argument vocabulary trait, returning its
// row, or nullptr if it is foreign or unrecognized. The module is pinned (root
// contains "extensibility", a qualifier "tensor_arg_traits") and the leaf
// matched exactly, so `Input` never matches `MutableInput`.
static const VocabEntry *classifyVocabTrait(TraitSymbolAttr traitSymbol) {
  mlir::SymbolRefAttr sym = traitSymbol.getSymbol();
  if (!sym.getRootReference().strref().contains("extensibility"))
    return nullptr;
  bool inModule = false;
  for (mlir::FlatSymbolRefAttr ref : sym.getNestedReferences()) {
    if (ref.getValue().contains("tensor_arg_traits")) {
      inModule = true;
      break;
    }
  }
  if (!inModule)
    return nullptr;
  for (const VocabEntry &entry : kTensorArgVocab)
    if (sym.getLeafReference() == entry.leaf)
      return &entry;
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelDeclFunctionAdaptor
//===----------------------------------------------------------------------===//

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    KGEN::LIT::FnTypeGeneratorType sig, mlir::ArrayAttr tensorArgContracts) {
  mlir::MLIRContext *ctx = sig.getContext();
  size_t numArgs = sig.getNumArguments();

  // The import-baked `tensor_arg_contracts` array (signature-arg order) is
  // authoritative; `bakedEntryFor` and `hasBakedArray` feed
  // `resolveTraitContract` via classifyArgType / countDPSOutputs.
  bool hasBakedArray = tensorArgContracts && !tensorArgContracts.empty();
  auto bakedEntryFor = [&](size_t i) -> mlir::Attribute {
    if (hasBakedArray && i < tensorArgContracts.size())
      return tensorArgContracts[i];
    return mlir::Attribute();
  };

  // Detect whether the result is passed as a trailing by-ref argument.
  bool resultAsArgument = false;
  for (size_t i = 0; i < numArgs; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult) {
      resultAsArgument = true;
      break;
    }
  }
  bool isThrow = sig.isThrows();
  size_t numByRefArgs = isThrow ? 2 : (resultAsArgument ? 1 : 0);
  size_t endOfInputArgs = numArgs - numByRefArgs;
  size_t numDPSOutputs = countDPSOutputs(sig, tensorArgContracts);

  // Output (DPS) arguments come first.
  for (size_t i = 0; i < numDPSOutputs; ++i) {
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(i), /*promoteSIMD=*/false,
                        bakedEntryFor(i), hasBakedArray);
    outputArguments.push_back(std::move(op));
  }

  // Input arguments follow.
  for (size_t i = numDPSOutputs; i < endOfInputArgs; ++i) {
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(i), /*promoteSIMD=*/false,
                        bakedEntryFor(i), hasBakedArray);
    inputArguments.push_back(std::move(op));
  }

  // Optional result: either passed by reference (last arg) or as a value.
  if (resultAsArgument && numByRefArgs > 0) {
    size_t byrefIdx = endOfInputArgs + numByRefArgs - 1;
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(byrefIdx), byrefIdx, /*isByRefResult=*/true};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(byrefIdx), /*promoteSIMD=*/false,
                        bakedEntryFor(byrefIdx), hasBakedArray);
    outputResult = std::move(op);
  } else if (!resultAsArgument) {
    mlir::Type resultType = sig.getResultType();
    if (!KGEN::sugarIsa<KGEN::NoneType>(resultType)) {
      KernelDeclOperandAdaptor op;
      // A value result isn't a signature argument, so the baked array doesn't
      // cover it; use the re-distill fallback.
      op.underlyingType =
          classifyArgType(ctx, resultType, /*promoteSIMD=*/false,
                          /*bakedEntry=*/mlir::Attribute(),
                          /*hasBakedArray=*/false);
      outputResult = std::move(op);
    }
  }
}

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    KernelFunctionAttr attr, StringAttr fullyResolvedName)
    : KernelDeclFunctionAdaptor(attr.getSignature(),
                                attr.getTensorArgContracts()) {
  this->attr = attr;
  this->fullyResolvedName = fullyResolvedName;
}

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    ShapeFunctionDeclOp shapeDecl)
    : KernelDeclFunctionAdaptor(shapeDecl.getFunction(),
                                shapeDecl.getSymNameAttr()) {}

KernelDeclFunctionAdaptor
KernelDeclFunctionAdaptor::synthesizeExecuteFromElementwise(
    KGEN::LIT::FnTypeGeneratorType sig) {
  mlir::MLIRContext *ctx = sig.getContext();
  KernelDeclFunctionAdaptor result;
  // Wrap the synthetic signature so `attr.getSignature()` works; params /
  // arg_params / result_params stay null.
  result.attr = KernelFunctionAttr::get(ctx, sig, /*params=*/{},
                                        /*arg_params=*/{},
                                        /*result_params=*/{},
                                        /*tensor_arg_contracts=*/{});

  size_t numArgs = sig.getNumArguments();
  bool resultAsArgument = false;
  for (size_t i = 0; i < numArgs; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult) {
      resultAsArgument = true;
      break;
    }
  }
  bool isThrow = sig.isThrows();
  size_t numByRefArgs = isThrow ? 2 : (resultAsArgument ? 1 : 0);
  size_t endOfInputArgs = numArgs - numByRefArgs;

  for (size_t i = 0; i < endOfInputArgs; ++i) {
    mlir::Type argType = sig.getArgument(i);

    // Skip a trailing IndexList argument (it is an elementwise-only concept).
    if (i + 1 == endOfInputArgs) {
      KGEN::LIT::StructType structType = getAsStructType(argType);
      if (structType && symbolMatchesRootAndLeaf(structType.getSymbol(),
                                                 kPackageStd, kLeafIndexList))
        continue;
    }

    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    // The synthetic elementwise->execute signature has no trait-bound args and
    // no baked contracts; use the re-distill fallback.
    op.underlyingType =
        classifyArgType(ctx, argType, /*promoteSIMDToFusedTensor=*/true,
                        /*bakedEntry=*/mlir::Attribute(),
                        /*hasBakedArray=*/false);
    result.inputArguments.push_back(std::move(op));
  }

  // Synthesise the single output tensor.
  KernelDeclOperandAdaptor outOp;
  outOp.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
      mlir::StringAttr::get(ctx, "out"), 0, /*isByRefResult=*/false};
  outOp.underlyingType = KernelDeclTensorAdaptor{IOSpec::FusedOutputTensor};
  result.outputArguments.push_back(std::move(outOp));

  return result;
}

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor
//===----------------------------------------------------------------------===//

KernelDeclAdaptor::KernelDeclAdaptor(KernelDeclOp decl) {
  mlir::MLIRContext *ctx = decl->getContext();
  registeredOpName = mlir::StringAttr::get(ctx, decl.getRegistration());
  fullyResolvedName = decl.getSymNameAttr();

  // Consumers read cached fn-level params and slot-name mappings through
  // `f.attr.*`.
  auto buildFn =
      [&](std::optional<KernelFunctionAttr> fnAttr,
          StringRef methodName) -> std::optional<KernelDeclFunctionAdaptor> {
    if (!fnAttr)
      return std::nullopt;
    KernelDeclFunctionAdaptor f(fnAttr->getSignature(),
                                fnAttr->getTensorArgContracts());
    f.attr = *fnAttr;
    f.fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.{}", fullyResolvedName, methodName).str());
    return f;
  };

  executeFunction =
      buildFn(decl.getExecute(),
              stringifyKernelFunctionKind(KernelFunctionKind::Execute));
  buildGraphFunction =
      buildFn(decl.getBuildGraph(),
              stringifyKernelFunctionKind(KernelFunctionKind::BuildGraph));
  elementwiseFunction =
      buildFn(decl.getElementwise(),
              stringifyKernelFunctionKind(KernelFunctionKind::Elementwise));
  // The TileTensor overload is a Mojo method named `elementwise` too.
  tileElementwiseFunction =
      buildFn(decl.getTileElementwise(),
              stringifyKernelFunctionKind(KernelFunctionKind::Elementwise));
  executeDeviceFunction =
      buildFn(decl.getExecuteDevice(),
              stringifyKernelFunctionKind(KernelFunctionKind::ExecuteDevice));
  executeConfigurationFunction = buildFn(
      decl.getExecuteConfiguration(),
      stringifyKernelFunctionKind(KernelFunctionKind::ExecuteConfiguration));

  if (executeDeviceFunction)
    executeFunction = executeDeviceFunction;

  if (elementwiseFunction && !executeFunction) {
    // When there is no explicit execute, synthesise one from the elementwise
    // signature so callers always have an executeFunction to query.
    executeFunction =
        KernelDeclFunctionAdaptor::synthesizeExecuteFromElementwise(
            decl.getElementwise()->getSignature());
  }

  if (KernelReductionAttr reductionAttr = decl.getReductionAttr())
    reduction = reductionAttr;
  structParams = decl.getStructParams();
  supportsUnfusedExecute = decl.getSupportsUnfusedExecute();
}

const KernelDeclFunctionAdaptor *
KernelDeclAdaptor::getFunction(KernelFunctionKind kind) const {
  auto get = [](const std::optional<KernelDeclFunctionAdaptor> &function) {
    return function ? &*function : nullptr;
  };
  switch (kind) {
  case KernelFunctionKind::Execute:
    return get(executeFunction);
  case KernelFunctionKind::Elementwise:
    return get(elementwiseFunction);
  case KernelFunctionKind::TileElementwise:
    return get(tileElementwiseFunction);
  case KernelFunctionKind::BuildGraph:
    return get(buildGraphFunction);
  case KernelFunctionKind::ExecuteDevice:
    return get(executeDeviceFunction);
  case KernelFunctionKind::ExecuteConfiguration:
    return get(executeConfigurationFunction);
  case KernelFunctionKind::Shape:
    return nullptr;
  }
  llvm_unreachable("unknown kernel function kind");
}

KernelFunctionDeclAdaptor::KernelFunctionDeclAdaptor(PrimitiveDeclOp decl)
    : registration(
          mlir::StringAttr::get(decl->getContext(), decl.getRegistration())),
      function(decl.getFunction().getSignature()) {
  function.fullyResolvedName = decl.getSymNameAttr();
}

//===----------------------------------------------------------------------===//
// IOSpec helpers.
//===----------------------------------------------------------------------===//

namespace M::MOGG {

StringRef toString(IOSpec spec) {
  switch (spec) {
  case IOSpec::InputTensor:
    return kInputTensor;
  case IOSpec::OutputTensor:
    return kOutputTensor;
  case IOSpec::MutableInputTensor:
    return kMutableInputTensor;
  case IOSpec::FusedInputTensor:
    return kFusedInputTensor;
  case IOSpec::FusedOutputTensor:
    return kFusedOutputTensor;
  case IOSpec::FusedComputeOutputTensor:
    return kFusedComputeOutputTensor;
  case IOSpec::FusedComputeOutputTileTensor:
    return kFusedComputeOutputTileTensor;
  case IOSpec::FusedOutputTileTensor:
    return kFusedOutputTileTensor;
  }
}

std::optional<IOSpec> toIOSpec(StringRef str) {
  if (str == kInputTensor)
    return IOSpec::InputTensor;
  if (str == kOutputTensor)
    return IOSpec::OutputTensor;
  if (str == kMutableInputTensor)
    return IOSpec::MutableInputTensor;
  if (str == kFusedInputTensor)
    return IOSpec::FusedInputTensor;
  if (str == kFusedOutputTensor)
    return IOSpec::FusedOutputTensor;
  if (str == kFusedComputeOutputTensor)
    return IOSpec::FusedComputeOutputTensor;
  if (str == kFusedComputeOutputTileTensor)
    return IOSpec::FusedComputeOutputTileTensor;
  if (str == kFusedOutputTileTensor)
    return IOSpec::FusedOutputTileTensor;
  return std::nullopt;
}

std::pair<TypedAttr, TypedAttr> getParams(IOSpec ioSpec, Builder &builder) {
  switch (ioSpec) {
  case IOSpec::InputTensor:
    return {builder.getBoolAttr(kIOSpecImmutable),
            builder.getIndexAttr(kIOSpecIOInput)};

  case IOSpec::OutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOOutput)};

  case IOSpec::MutableInputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOInput)};

  case IOSpec::FusedInputTensor:
    return {builder.getBoolAttr(kIOSpecImmutable),
            builder.getIndexAttr(kIOSpecIOFusedInput)};

  case IOSpec::FusedOutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedOutput)};

  case IOSpec::FusedComputeOutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedComputeOutput)};

  case IOSpec::FusedComputeOutputTileTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedComputeOutputTile)};

  case IOSpec::FusedOutputTileTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedOutputTile)};
  }
}

bool isOutputIOSpec(IOSpec spec) {
  return spec == IOSpec::OutputTensor || spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor ||
         spec == IOSpec::FusedComputeOutputTileTensor ||
         spec == IOSpec::FusedOutputTileTensor;
}

bool isFusableIOSpec(IOSpec spec) {
  return spec == IOSpec::FusedInputTensor ||
         spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor ||
         spec == IOSpec::FusedComputeOutputTileTensor ||
         spec == IOSpec::FusedOutputTileTensor;
}

bool isMutableIOSpec(IOSpec spec) { return spec == IOSpec::MutableInputTensor; }

mlir::FailureOr<TensorArgContract> distillTensorArgContract(
    llvm::ArrayRef<M::KGEN::TraitSymbolAttr> writtenSymbols,
    llvm::function_ref<void(const llvm::Twine &)> emitError, bool isShapeFunc) {
  // Per-row presence over the vocabulary table. Presence is idempotent, so a
  // trait surfacing more than once (e.g. both as written and as another
  // trait's ancestor) counts once.
  bool present[std::size(kTensorArgVocab)] = {};
  for (TraitSymbolAttr traitSymbol : writtenSymbols) {
    if (const VocabEntry *entry = classifyVocabTrait(traitSymbol)) {
      present[entry - kTensorArgVocab] = true;
      continue;
    }
    // Not a vocabulary trait. Canonicalization folds a composition into its
    // ancestor closure, and the only non-vocabulary ancestor of a vocabulary
    // trait is the universal `AnyType` root (`TileTensorable: Tensor`,
    // `Loadable: LoadAccess`, ...). Accept `AnyType`; anything else is a
    // foreign trait the author wrote explicitly.
    mlir::SymbolRefAttr sym = traitSymbol.getSymbol();
    if (sym.getLeafReference() == "AnyType")
      continue;
    emitError("trait-bound tensor argument references '" +
              sym.getLeafReference().getValue() +
              "', which is not a tensor-argument trait; only traits from "
              "extensibility.tensor_arg_traits are allowed");
    return mlir::failure();
  }

  // Visit the present vocabulary rows of one category, in table order.
  auto forEachPresent = [&](TraitCategory category,
                            llvm::function_ref<void(const VocabEntry &)> fn) {
    for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
      if (present[i] && kTensorArgVocab[i].category == category)
        fn(kTensorArgVocab[i]);
  };
  auto anyPresent = [&](TraitCategory category) {
    bool any = false;
    forEachPresent(category, [&](const VocabEntry &) { any = true; });
    return any;
  };

  auto isPresentLeaf = [&](llvm::StringRef leaf) {
    for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
      if (kTensorArgVocab[i].leaf == leaf)
        return present[i];
    return false;
  };

  // Shape-function lane: arguments always run on the host and are
  // read-only, so reading through `TileTensorable` is safe here (unlike
  // the execute lane). The distiller records the deepest bound written
  // (`Tensor` or `TileTensorable`) in `contract.bound`.
  if (isShapeFunc) {
    if (anyPresent(TraitCategory::role)) {
      emitError("shape-function tensor argument carries an io-role marker "
                "(Input, Output, or MutableInput). Shape-function arguments "
                "are read-only. They must carry no role. Conform to Tensor "
                "or TileTensorable instead");
      return mlir::failure();
    }
    if (anyPresent(TraitCategory::fused)) {
      emitError("Fused is not allowed on a shape-function tensor argument");
      return mlir::failure();
    }
    if (anyPresent(TraitCategory::storage)) {
      emitError("Stable is not allowed on a shape-function tensor argument; "
                "shape functions run on the host and cannot read a stable "
                "address slot");
      return mlir::failure();
    }
    if (anyPresent(TraitCategory::accessMarker) ||
        anyPresent(TraitCategory::accessCapability) ||
        anyPresent(TraitCategory::deferred)) {
      emitError(
          "LoadAccess, Loadable, StoreAccess, Storable, _LegacyComputeOutput, "
          "and _LegacyComputable are fusion machinery. They are not allowed "
          "on a shape-function tensor argument. Conform to Tensor or "
          "TileTensorable instead");
      return mlir::failure();
    }
    TensorArgContract contract;
    contract.role = TensorArgRole::none;
    contract.fused = false;
    if (isPresentLeaf(kTileTensorableLeaf))
      contract.bound = TensorArgBound::tile;
    else
      contract.bound = TensorArgBound::tensor;
    return contract;
  }

  // Exactly one role trait.
  int roleCount = 0;
  TensorArgRole role = TensorArgRole::input;
  forEachPresent(TraitCategory::role, [&](const VocabEntry &entry) {
    ++roleCount;
    role = entry.role;
  });
  if (roleCount == 0) {
    emitError("trait-bound tensor argument has no role trait (expected exactly "
              "one of Input, Output, MutableInput)");
    return mlir::failure();
  }
  if (roleCount > 1) {
    emitError("trait-bound tensor argument has more than one role trait "
              "(expected exactly one of Input, Output, MutableInput)");
    return mlir::failure();
  }

  TensorArgContract contract;
  contract.role = role;

  contract.fused = anyPresent(TraitCategory::fused);

  // Storage: a `Stable` argument is read through a `TileTensor` projection
  // over its slot, so it cannot be a fusion functor or a pointer-written
  // output.
  contract.stable = anyPresent(TraitCategory::storage);
  if (contract.stable) {
    if (contract.fused) {
      emitError("a Stable tensor argument cannot also be Fused; a stable "
                "address slot is read through a TileTensor projection, not a "
                "fusion functor");
      return mlir::failure();
    }
    if (contract.role == TensorArgRole::output) {
      emitError("Stable is not valid on an Output argument; outputs are "
                "written through a pointer-backed buffer (use Input or "
                "MutableInput)");
      return mlir::failure();
    }
  }

  // Tile-transform fusion is deferred.
  if (anyPresent(TraitCategory::deferred)) {
    emitError("tile-transform fusion is not supported yet");
    return mlir::failure();
  }

  // Access forms are a set, keyed by marker or capability leaf: a composition
  // may carry only the capability (e.g. `Loadable`) or only the marker
  // (opaque-device host descriptor); the capability bit tracks the callable
  // leaf. A non-fused arg may carry several (e.g. `Loadable & Storable` for
  // in-place read-modify-write).
  forEachPresent(TraitCategory::accessMarker, [&](const VocabEntry &entry) {
    contract.accessMask |= bit(entry.access);
  });
  forEachPresent(TraitCategory::accessCapability, [&](const VocabEntry &entry) {
    contract.accessMask |= bit(entry.access);
    contract.capabilityMask |= bit(entry.access);
  });

  // A fused argument must declare exactly one access form.
  if (contract.fused && contract.accessCount() != 1) {
    emitError("a Fused tensor argument must declare exactly one access form "
              "(one of LoadAccess/Loadable, StoreAccess/Storable, or "
              "_LegacyComputeOutput)");
    return mlir::failure();
  }

  // Role / access compatibility, applied per present form. Multiple forms are
  // allowed on a non-fused arg, so each is checked independently:
  //   * `load`      invalid on Output.
  //   * `store`     invalid on a read-only Input.
  //   * `transform` requires Output.
  // (So `load & store` together implies a MutableInput.)
  if (contract.hasAccess(TensorArgAccess::load) &&
      contract.role == TensorArgRole::output) {
    emitError("load access is not valid on an Output argument");
    return mlir::failure();
  }
  if (contract.hasAccess(TensorArgAccess::store) &&
      contract.role == TensorArgRole::input) {
    emitError("store access is not valid on a read-only Input argument");
    return mlir::failure();
  }
  if (contract.hasAccess(TensorArgAccess::transform) &&
      contract.role != TensorArgRole::output) {
    emitError("transform access requires an Output argument");
    return mlir::failure();
  }

  // The graph compiler may deliver a `_LegacyComputeOutput` slot as a store
  // fusion, so a signature can't demand the compute capability up front.
  if (contract.hasCapability(TensorArgAccess::transform)) {
    emitError("declare _LegacyComputeOutput on the kernel signature and "
              "refine to _LegacyComputable in the body; the graph compiler "
              "may deliver a store fusion instead");
    return mlir::failure();
  }
  // The store-fusion wrapper doesn't conform to `TileTensorable`.
  if (contract.hasAccess(TensorArgAccess::transform) &&
      isPresentLeaf(kTileTensorableLeaf)) {
    emitError("a _LegacyComputeOutput argument cannot require TileTensorable "
              "on the kernel signature; refine in the body, since the graph "
              "compiler may deliver a store fusion instead");
    return mlir::failure();
  }

  return contract;
}

KGEN::LIT::StructType getLegacyTensorArgType(mlir::Type argType) {
  KGEN::LIT::StructType structType = getAsStructType(argType);
  if (!structType)
    return {};
  auto sym = structType.getSymbol();
  if (symbolMatches(sym, kMaxManagedTensorSlice) ||
      symbolMatches(sym, kMaxVariadicTensors) ||
      symbolMatches(sym, kMaxFusedInputVariadicTensors) ||
      symbolMatches(sym, kMaxFusedOutputVariadicTensors) ||
      symbolMatches(sym, kMaxStableTensor) ||
      symbolMatches(sym, kMaxStableVariadicTensors))
    return structType;
  return {};
}

KGEN::LIT::TraitType getTensorArgTraitType(mlir::Type argType) {
  mlir::Type t = argType;
  // Borrowed tensor args are wrapped in a ref; strip it (mirrors
  // getAsStructType).
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(t))
    t = ref.getElementType();
  // A trait-bound arg `x: T` is a reference to the generic type parameter `T`,
  // whose metatype is the written `!lit.trait<...>` bound. Mirrors the
  // type-parameter branch of `simplifyConformsToAgainstTypeValue`.
  auto paramType = KGEN::sugarDynCast<KGEN::ParamType>(t);
  if (!paramType)
    return {};
  auto traitType = mlir::dyn_cast_if_present<KGEN::LIT::TraitType>(
      KGEN::getCanonicalType(paramType.getMetaType()));
  if (!traitType)
    return {};
  // A generic arg bound only by other traits (e.g. a primitive's
  // `[T: ImplicitlyCopyable]`) is not a tensor arg.
  for (TraitSymbolAttr sym : traitType.getSymbols())
    if (classifyVocabTrait(sym))
      return traitType;
  return {};
}

KGEN::LIT::TraitType getTensorArgPackTraitType(mlir::Type argType) {
  KGEN::LIT::StructType pack = getAsStructType(argType);
  if (!pack || !symbolMatchesRootAndLeaf(pack.getSymbol(), kPackageStd,
                                         kLeafVariadicPack))
    return {};
  // The pack's `element_trait` parameter is the `*Ts` bound, carried as a
  // type-valued parameter. Take the first one that is a tensor-argument bound.
  for (mlir::Attribute param : pack.getParamValues()) {
    auto typeParam = mlir::dyn_cast<KGEN::TypeParamAttr>(
        KGEN::getCanonicalAttr(mlir::cast<mlir::TypedAttr>(param)));
    if (!typeParam)
      continue;
    auto traitType = mlir::dyn_cast_if_present<KGEN::LIT::TraitType>(
        KGEN::getCanonicalType(typeParam.getTypeValue()));
    if (!traitType)
      continue;
    for (TraitSymbolAttr sym : traitType.getSymbols())
      if (classifyVocabTrait(sym))
        return traitType;
  }
  return {};
}

KGEN::LIT::TraitType getTensorArgTensorTupleTraitType(mlir::Type argType) {
  KGEN::LIT::StructType tuple = getAsStructType(argType);
  if (!tuple || !symbolMatchesRootAndLeaf(
                    tuple.getSymbol(), kPackageExtensibility, kLeafTensorTuple))
    return {};
  // `TensorTuple[*Ts]` carries the kernel's `Ts` list upcast to the tuple's own
  // `AnyType` element bound. Under the upcast, the list is typed by the
  // kernel's bound.
  for (mlir::Attribute param : tuple.getParamValues()) {
    auto value = mlir::dyn_cast<mlir::TypedAttr>(param);
    if (!value)
      continue;
    auto list = mlir::dyn_cast<KGEN::ParamListType>(
        KGEN::UpcastAttr::strip(KGEN::getCanonicalAttr(value)).getType());
    if (!list)
      continue;
    auto traitType = mlir::dyn_cast_if_present<KGEN::LIT::TraitType>(
        KGEN::getCanonicalType(list.getElementType()));
    if (!traitType)
      continue;
    for (TraitSymbolAttr sym : traitType.getSymbols())
      if (classifyVocabTrait(sym))
        return traitType;
  }
  return {};
}

llvm::SmallVector<TraitSymbolAttr>
getTensorArgTraitSymbols(mlir::Type argType) {
  KGEN::LIT::TraitType traitType = getTensorArgTraitType(argType);
  if (!traitType)
    traitType = getTensorArgTensorTupleTraitType(argType);
  if (!traitType)
    return {};
  return llvm::to_vector(traitType.getSymbols());
}

StringRef toString(TensorArgRole role) {
  switch (role) {
  case TensorArgRole::input:
    return "input";
  case TensorArgRole::output:
    return "output";
  case TensorArgRole::mutableInput:
    return "mutable_input";
  case TensorArgRole::none:
    return "none";
  }
  llvm_unreachable("bad TensorArgRole");
}

StringRef toString(TensorArgAccess access) {
  switch (access) {
  case TensorArgAccess::none:
    return "none";
  case TensorArgAccess::load:
    return "load";
  case TensorArgAccess::store:
    return "store";
  case TensorArgAccess::transform:
    return "transform";
  }
  llvm_unreachable("bad TensorArgAccess");
}

StringRef toString(TensorArgBound bound) {
  switch (bound) {
  case TensorArgBound::unspecified:
    return "unspecified";
  case TensorArgBound::tensor:
    return "tensor";
  case TensorArgBound::tile:
    return "tile";
  }
  llvm_unreachable("bad TensorArgBound");
}

mlir::DictionaryAttr
tensorArgContractToDict(mlir::MLIRContext *ctx,
                        const TensorArgContract &contract) {
  mlir::Builder b(ctx);
  // The access/capability SETS are encoded as integer bitmasks (the same bit
  // layout as `bit(TensorArgAccess)`); `dictToTensorArgContract` round-trips
  // them directly.
  llvm::SmallVector<mlir::NamedAttribute> entries{
      b.getNamedAttr(kRoleKey, b.getStringAttr(toString(contract.role))),
      b.getNamedAttr(kFusedKey, b.getBoolAttr(contract.fused)),
      b.getNamedAttr(kAccessKey, b.getI32IntegerAttr(static_cast<int32_t>(
                                     contract.accessMask))),
      b.getNamedAttr(kCapabilitiesKey, b.getI32IntegerAttr(static_cast<int32_t>(
                                           contract.capabilityMask)))};
  // Omitted when unspecified, keeping execute-lane encodings byte-identical.
  if (contract.bound != TensorArgBound::unspecified)
    entries.push_back(
        b.getNamedAttr(kBoundKey, b.getStringAttr(toString(contract.bound))));
  // Likewise `stable` is omitted when false.
  if (contract.stable)
    entries.push_back(b.getNamedAttr(kStableKey, b.getBoolAttr(true)));
  return mlir::DictionaryAttr::get(ctx, entries);
}

MOGG::TensorArgContractAttr
tensorArgContractToAttr(mlir::MLIRContext *ctx,
                        const TensorArgContract &contract) {
  // Map the C++ contract enums to the IR enums. The two enums are kept in
  // lockstep (same case names) -- see MOGGEnums.td.
  TensorArgRole role = contract.role;
  MOGG::TensorArgRoleKind irRole;
  switch (role) {
  case TensorArgRole::input:
    irRole = MOGG::TensorArgRoleKind::input;
    break;
  case TensorArgRole::output:
    irRole = MOGG::TensorArgRoleKind::output;
    break;
  case TensorArgRole::mutableInput:
    irRole = MOGG::TensorArgRoleKind::mutable_input;
    break;
  case TensorArgRole::none:
    irRole = MOGG::TensorArgRoleKind::none;
    break;
  }

  // The access/capability bitmasks share the bit layout of the IR bit enum
  // (`bit(TensorArgAccess)` == `TensorArgAccessKind` bits), so cast directly.
  auto irAccess = static_cast<MOGG::TensorArgAccessKind>(contract.accessMask);
  auto irCapabilities =
      static_cast<MOGG::TensorArgAccessKind>(contract.capabilityMask);

  // Same lockstep mapping as `irRole` above.
  MOGG::TensorArgBoundKind irBound;
  switch (contract.bound) {
  case TensorArgBound::unspecified:
    irBound = MOGG::TensorArgBoundKind::unspecified;
    break;
  case TensorArgBound::tensor:
    irBound = MOGG::TensorArgBoundKind::tensor;
    break;
  case TensorArgBound::tile:
    irBound = MOGG::TensorArgBoundKind::tile;
    break;
  }

  return MOGG::TensorArgContractAttr::get(ctx, irRole, contract.fused, irAccess,
                                          irCapabilities, irBound,
                                          contract.stable);
}

std::optional<TensorArgContract> dictToTensorArgContract(mlir::Attribute attr) {
  // The "no contract" sentinel: a non-trait argument is baked as a UnitAttr
  // placeholder (and a null attr is treated the same way).
  auto dict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(attr);
  if (!dict)
    return std::nullopt;

  auto getStr = [&](StringRef key) -> std::optional<StringRef> {
    if (auto s = dict.getAs<mlir::StringAttr>(key))
      return s.strref();
    return std::nullopt;
  };
  auto getBool = [&](StringRef key) -> std::optional<bool> {
    if (auto b = dict.getAs<mlir::BoolAttr>(key))
      return b.getValue();
    return std::nullopt;
  };
  auto getMask = [&](StringRef key) -> std::optional<unsigned> {
    if (auto i = dict.getAs<mlir::IntegerAttr>(key))
      return static_cast<unsigned>(i.getInt());
    return std::nullopt;
  };

  std::optional<StringRef> roleStr = getStr(kRoleKey);
  std::optional<bool> fused = getBool(kFusedKey);
  std::optional<unsigned> accessMask = getMask(kAccessKey);
  std::optional<unsigned> capabilityMask = getMask(kCapabilitiesKey);
  if (!roleStr || !fused || !accessMask || !capabilityMask)
    return std::nullopt;
  // `bound` is only ever baked for a shape-fn contract; absent means
  // `unspecified` (every execute-lane encoding).
  std::optional<StringRef> boundStr = getStr(kBoundKey);
  // `stable` is only ever baked when set; absent means false.
  std::optional<bool> stable = getBool(kStableKey);

  // Inverse of `toString(TensorArgRole)`.
  TensorArgRole role;
  if (*roleStr == toString(TensorArgRole::input))
    role = TensorArgRole::input;
  else if (*roleStr == toString(TensorArgRole::output))
    role = TensorArgRole::output;
  else if (*roleStr == toString(TensorArgRole::mutableInput))
    role = TensorArgRole::mutableInput;
  else if (*roleStr == toString(TensorArgRole::none))
    role = TensorArgRole::none;
  else
    return std::nullopt;

  // Inverse of `toString(TensorArgBound)`.
  TensorArgBound bound = TensorArgBound::unspecified;
  if (boundStr) {
    if (*boundStr == toString(TensorArgBound::tensor))
      bound = TensorArgBound::tensor;
    else if (*boundStr == toString(TensorArgBound::tile))
      bound = TensorArgBound::tile;
    else if (*boundStr != toString(TensorArgBound::unspecified))
      return std::nullopt;
  }

  TensorArgContract contract;
  contract.role = role;
  contract.fused = *fused;
  contract.accessMask = *accessMask;
  contract.capabilityMask = *capabilityMask;
  contract.bound = bound;
  contract.stable = stable.value_or(false);
  return contract;
}

} // namespace M::MOGG
