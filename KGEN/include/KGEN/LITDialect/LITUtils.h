//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions primarily for parsing, printing and
// verifying LIT related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITUTILS_H
#define KGEN_LITDIALECT_LITUTILS_H

#include "KGEN/KGENDialect/KGENPogUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
class SymbolOpInterface;
class SymbolTable;
} // namespace mlir

namespace M {
class StringArrayAttr;
template <typename T>
class ErrorOr;

namespace KGEN {
class FnEffects;
class ParamDeclAttr;
class ParamDeclArrayAttr;
class ParameterEvaluator;
class ParameterExprArrayAttr;
enum class ArgConvention : uint32_t;
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;

namespace LIT {
class FnType;

/// Returns whether the given type is a LIT meta type.
bool isMetaType(Type type);
bool isVariadicOfMetaType(Type type);

/// Returns whether the given attribute is a (variadic of) LIT type expression.
bool isTypeExpr(TypedAttr attr);
bool isVariadicOfTypeExpr(TypedAttr attr);

/// Returns whether the given attribute is a LIT level-1 type expression (e.g.,
/// a type expression that describe a struct type, such as struct meta type and
/// trait type).
bool isFirstLevelTypeExpr(TypedAttr attr);

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print/Parse a (potentially) parametric mutability specifier and then a
/// value.  The three forms are: "imm expr", "mut expr", "mut=<expr>, expr"
/// without quotes.
ParseResult parseOriginParamValue(AsmParser &p, TypedAttr &result);
void printOriginParamValue(AsmPrinter &p, TypedAttr value);
inline void printOriginParamValue(AsmPrinter &p, Operation *, TypedAttr value) {
  printOriginParamValue(p, value);
}

/// Pretty print a nested symbol reference to a name.
void printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol);

/// Parse an optional default value of the given type, with reference-type
/// unwrapping. `defaultVal` is not modified if a default value was not
/// present. If `hasAddress` is set, the default value is parsed as if `type`
/// is an address type: either a pointer or reference. The method is tolerant
/// if `type` is not actually one.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type, bool hasAddress);
void printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal, Type type,
                               bool hasAddress);

/// Parse and print a origin set.
ParseResult parseOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
OptionalParseResult
parseOptionalOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
void printOriginSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes);

/// Return true if the origin set parameter is an empty set.
bool isEmptyOriginSet(TypedAttr attr);

void printFnType(AsmPrinter &p, FnType signature);

//===----------------------------------------------------------------------===//
// MangledSymbol
//===----------------------------------------------------------------------===//

/// This class provides a wrapper around a mojo FuncOp that mangles its name (in
/// `mangled`) but also provides all the components of the mangled name. If the
/// func is already mangled, this will pull everything apart.
struct MangledSymbol {
  /// Mangle the symbol for this op by walking upwards and adding struct/module
  /// names.
  static MangledSymbol mangle(mlir::SymbolOpInterface op);
  /// Demangle this mangled name by parsing it into its component parts.
  static FailureOr<MangledSymbol> demangle(StringAttr mangled,
                                           bool parseSignature = true);

  /// The format for a mangled name is roughly:
  ///  $<module name>::<struct name>[::<struct name>]
  ///    ::<function name>[<comma separated params>]
  ///      (<comma-separated args>)<comma-separated results>

  /// The fully mangled name.
  StringAttr mangled;
  /// The various strings that make up the mangled name.
  SmallVector<StringAttr, 1> moduleNames;
  /// We support nested structs, so there may be more than one struct name.
  SmallVector<StringAttr, 1> structNames;
  /// The bare name of the symbol, which may include parameters.
  StringAttr symName;
  /// The bare name of the symbol without parameters.
  StringAttr identifier;
  /// If the symbol has a signature mangled into the name, then it will be here.
  FunctionType signature;
};

/// Print a mangled symbol.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MangledSymbol &ms);

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

/// Verify the the order of passing kinds, and that the number of defaults
/// doesn't exceed the number of corresponding passing kinds.
LogicalResult verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<PogMetadataAttr> pogs,
                                 StringRef argOrParam);

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

void sortAndDeduplicateSymbols(SmallVectorImpl<SymbolRefAttr> &symbols);
void canonicalizeTraitCompositionSymbols(
    SmallVectorImpl<SymbolRefAttr> &symbols,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// Simplify a conforms_to attr by checking if the type value's trait bounds
/// already prove conformance. Extracts the TraitType from the type value
/// automatically, handling both parser and post-lower-lit representations.
FailureOr<TypedAttr> simplifyConformsToAgainstTypeValue(
    TypeConformsToTraitAttr conformsTo,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// Fold a DowncastAttr when its input is a concrete LIT struct type value.
/// Returns the folded TypeParamAttr, or null if the downcast can't be folded.
TypedAttr foldDowncastToStructType(DowncastAttr downcast);

/// LIT dialect evaluation context. Resolves LIT struct declarations for struct
/// reflection operations. Inherits common dispatch from base class.
class LITSymTabEvaluationContext : public SymTabEvaluationContext {
public:
  using SymTabEvaluationContext::SymTabEvaluationContext;

protected:
  /// Resolve struct info for LIT dialect structs with fallback to KGEN.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Resolve a function symbol via the LIT/KGEN symbol table. Returns the
  /// `lit.fn` op if present, otherwise falls back to the kgen-generator
  /// lookup inherited from `SymTabEvaluationContext`.
  FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) override;

  /// Resolve conformance using the struct's symbol table.
  Operation *resolveConformanceForStruct(ResolvedStructHandle resolved,
                                         StringAttr traitName) override;

  /// Handle LIT-specific attributes (Downcast, TypeConformsToTrait).
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;
};

//===----------------------------------------------------------------------===//
// IndexToDeclRefRemapper
//===----------------------------------------------------------------------===//

/// Utility class for remapping index references to parameter declaration
/// references using metadata from a PogListAttr.
class IndexToDeclRefRemapper
    : public IndexParameterReplacer<IndexToDeclRefRemapper> {
public:
  IndexToDeclRefRemapper(PogListAttr paramListAttr)
      : paramListAttr(paramListAttr) {}

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexToDeclRefRemapper>;

  PogListAttr paramListAttr;
};

//===----------------------------------------------------------------------===//
// Constraint Checking
//===----------------------------------------------------------------------===//

/// The logical relationship between an assumption and a proposition.
enum class ConstraintRelation {
  Implies,     ///< assumption implies proposition.
  Contradicts, ///< assumption implies NOT proposition.
  Unprovable,  ///< neither relation is provable.
};

/// Determine how propA (an assumption) relates to propB (a proposition).
/// Returns Implies, Contradicts, or Unprovable.
ConstraintRelation inferConstraintRelation(TypedAttr propA, TypedAttr propB);

/// Returns true if propA implies propB. Thin wrapper over
/// inferConstraintRelation.
inline bool constraintImplies(TypedAttr propA, TypedAttr propB) {
  return inferConstraintRelation(propA, propB) == ConstraintRelation::Implies;
}

/// Result of checking whether a type conforms to a trait.
/// This is a 3-state result because conditional conformances may not be
/// provable at compile time.
enum class ConformanceResult {
  Yes,          // Definitely conforms (unconditional or constraint proven true)
  No,           // Definitely does not conform (constraint proven false)
  NeedsEvidence // Conditional conformance that can't be proven statically
};

/// Evaluate a conditional constraint using a pre-built evaluator that already
/// has struct parameters bound. Caller assumptions may prove or disprove the
/// rebound constraint via inferConstraintRelation.
ConformanceResult
evaluateConstraint(ParameterEvaluator &evaluator, ConstraintAttr constraint,
                   ArrayRef<ConstraintAttr> callerAssumptions = {});

/// Given a type expression and a set of assumptions, compute the effective
/// trait bound implied by any `conforms_to(type, Trait)` constraints. Returns a
/// null TraitType if no refinements apply.
TraitType getTraitBoundFromAssumptions(
    TypedAttr typeAttr, ArrayRef<ConstraintAttr> assumptions,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver);

/// TODO: `ClosureEmitter.cpp` has a nearly identical helper
/// (`getUnderlyingParamRef`). Unify these implementations to avoid drift.
///
/// Extract the underlying ParamDeclRefAttr from a type expression by peeling
/// UpcastAttr and TypeParamAttr/ParamType wrappers.
/// Returns a null attr if no ParamDeclRefAttr can be extracted.
ParamDeclRefAttr extractParamDeclRef(TypedAttr attr);

} // namespace LIT

//===----------------------------------------------------------------------===//
// Mojo identifier strings consumed by the Mojo parser.
//
// These are the canonical home for Mojo-side identifier names that both KGEN
// and the GraphCompiler need to agree on. GraphCompiler's `MojoIdentifiers.h`
// re-exports them from here so downstream callers keep using a single
// namespace; do not duplicate definitions on the GraphCompiler side.
//===----------------------------------------------------------------------===//

/// MLIR attribute keys carried on kernel `LIT::FnOp`s. The value of each is
/// a `DictionaryAttr` mapping a trait method name (e.g. `__del__`) to the
/// (potentially unresolved) method reference as a TypedAttr.
constexpr StringLiteral kMoggArgumentValueWitnesses =
    "mogg.arg_value_witnesses";
constexpr StringLiteral kMoggResultValueWitnesses =
    "mogg.result_value_witnesses";

/// Mojo source-level decorator and method names.
constexpr StringLiteral kFnRegisterInternal = "register_internal";
constexpr StringLiteral kFnRegister = "register";
constexpr StringLiteral kMoggExecuteFuncName = "execute";

/// Mojo package and type leaf names used to recognise extensibility kernel
/// types in MLIR symbol references.
constexpr StringLiteral kPackageStd = "std";
constexpr StringLiteral kPackageExtensibility = "extensibility";
constexpr StringLiteral kLeafManagedTensorSlice = "ManagedTensorSlice";
constexpr StringLiteral kLeafDeviceContext = "DeviceContext";

/// True iff `structTy` is the DPS `ManagedTensorSlice` from the
/// `extensibility` package (used by graph-compiler kernels).
inline bool isDPSTensor(LIT::StructType structTy) {
  return structTy.getSymbol().getRootReference().strref().starts_with(
             kPackageExtensibility) &&
         structTy.getSymbol().getLeafReference() == kLeafManagedTensorSlice;
}

/// True iff `structTy` is the public `std::DeviceContext` kernel-launch type.
inline bool isMojoDeviceContext(LIT::StructType structTy) {
  return structTy.getSymbol().getRootReference() == kPackageStd &&
         structTy.getSymbol().getLeafReference() == kLeafDeviceContext;
}

inline bool fnNeedsConformances(LIT::FnOp fnOp) {
  return fnOp.getSourceName() == kMoggExecuteFuncName;
}

} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
