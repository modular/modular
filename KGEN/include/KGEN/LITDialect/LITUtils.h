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

namespace LIT {
class FnType;
enum class PassingKind : uint32_t;

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
// Parameter Mangling
//===----------------------------------------------------------------------===//

/// Demangle a mangled parameter name if it is has a "`" postfix and and
/// trailing depth and unique ID. If `forUser` is true, then any prefixes for
/// autoparameters are removed.  If not, only the `42 suffix is removed.  The
/// later is important when calculating the ASTDecl name for a parameter.  The
/// former is useful when printing the name.
StringRef demangleParameterName(StringRef name, bool forUser = false);

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

/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present. If `hasAddress` is set, the
/// default value is parsed as if `type` is an address type: either a pointer or
/// reference. The method is tolerant if `type` is not actually one.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type, bool hasAddress = false);
void printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal, Type type,
                               bool hasAddress = false);

/// Parse a parameter specification in a lit op.
ParseResult parseOptionalParameterSpec(AsmParser &p,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       PogListAttr &paramListAttr);

/// Print a parameter specification in a lit op. A ParameterEvaluator is
/// necessary to substitute parameters into parametric parameters.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> paramDecls,
                                PogListAttr paramListAttr,
                                ParameterEvaluator &evaluator);

/// Parse a list of constraint attributes.
ParseResult
parseOptionalConstraintsList(AsmParser &p,
                             SmallVectorImpl<ConstraintAttr> &constraints);

/// Print a list of constraint attributes.
void printOptionalConstraintsList(AsmPrinter &p,
                                  ArrayRef<ConstraintAttr> constraints,
                                  ParameterEvaluator *evaluator = nullptr);

/// Parse an optional 'where' clause containing constraint attributes.
ParseResult
parseOptionalWhereClauses(AsmParser &p,
                          SmallVectorImpl<ConstraintAttr> &constraints);

/// Print an optional 'where' clause containing constraint attributes. The
/// ParameterEvaluator is used to rebind references in the constraints.
void printOptionalWhereClauses(AsmPrinter &p,
                               ArrayRef<ConstraintAttr> constraints,
                               ParameterEvaluator *evaluator = nullptr);

/// Parse a parameter signature (input/result types with optional default
/// values) if present. If `parseBody` is provided, it will be called after
/// parsing the input parameter spec.
ParseResult parseOptionalParamSignature(AsmParser &p,
                                        SmallVectorImpl<Type> &inputParamTypes,
                                        PogListAttr &paramListAttr,
                                        function_ref<ParseResult()> parseBody);

/// Print the parameter type signature if there are any input or result types,
/// along with the default input parameter values.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 PogListAttr paramListAttr);

/// Parse an optional parameter or argument name.
ParseResult parseOptionalName(AsmParser &p, StringAttr &name);

/// Parse an optional passing convention and variadicness. The the given index
/// will be added to the appropriate index array if a variadicness is present.
ParseResult parseConventionAndVariadicness(
    AsmParser &p, ArgConvention &convention, VariadicKind &variadic,
    std::optional<ArgConvention> &origVariadicConvention, size_t idx);

/// Print an optional passing convention and variadicness.
void printConventionAndVariadicness(AsmPrinter &p, ArgConvention convention,
                                    VariadicKind variadicness);

/// Parse and print a origin set.
ParseResult parseOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
OptionalParseResult
parseOptionalOriginSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
void printOriginSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes);

/// Return true if the origin set parameter is an empty set.
bool isEmptyOriginSet(TypedAttr attr);

void printFnType(AsmPrinter &p, FnType signature);

//===----------------------------------------------------------------------===//
// Pog Utils
//===----------------------------------------------------------------------===//

/// Count the number of inferred passing kinds.
size_t countNumInferredKinds(ArrayRef<PogMetadataAttr> pogs);
size_t countNumInferredKinds(PogListAttr pogListAttr);

/// Count the number of positional-only passing kinds.
size_t countNumPosOnly(ArrayRef<PogMetadataAttr> pogs);
size_t countNumPosOnly(PogListAttr pogListAttr);

/// Count the number of positional (pos-only or pos-or-kw) passing kinds.
size_t countNumPositional(ArrayRef<PogMetadataAttr> pogs);
size_t countNumPositional(PogListAttr pogListAttr);

/// Count the number of implicit passing kinds.
size_t countNumImplicitKinds(ArrayRef<PogMetadataAttr> pogs);
size_t countNumImplicitKinds(PogListAttr pogListAttr);

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

/// Handles parsing '|' and '*' in lit IR and counts the number of arguments of
/// different passing kinds.
/// TODO(#23387): fix this when AsmParser can handle '/'.
class PassingKindParser {
public:
  enum Marker { PLUS, BAR, STAR, QUESTION, NUM_MARKERS };
  static constexpr std::array<char, NUM_MARKERS> markers{'+', '|', '*', '?'};

  PassingKindParser(AsmParser &parser) : parser(parser) {}

  /// Try to parse a single optional '*' or '|', and emit an error if a
  /// duplicate is found or a '|' comes after a '*'.
  OptionalParseResult parseOptionalStarSlash();

  /// Populate the parameter passing kinds.
  void populatePassingKinds(SmallVectorImpl<PassingKind> &kinds) const;

  /// Return true if the parser is currently parsing an implicit parameter.
  bool isCurrentImplicit() const { return foundMarkers[QUESTION]; }

  /// Return true if the parser is currently parsing a keyword-only parameter.
  bool isCurrentKwOnly() const {
    return foundMarkers[STAR] && !foundMarkers[QUESTION];
  }

private:
  AsmParser &parser;
  size_t idx = 0;
  std::array<bool, NUM_MARKERS> foundMarkers{};
  std::array<size_t, NUM_MARKERS> idxOfEach{};
};

/// Handles printing '/', '+', '?', and '*' in lit IR. Optionally, it allows
/// specifying a replacement to be used instead of '/' and '+'. It also allows
/// specifying a flag to suppress the '/' if it immediately follows the first
/// argument (useful if printing methods with mojo syntax).
class PassingKindPrinter {
public:
  PassingKindPrinter(raw_ostream &os, size_t numPogs,
                     std::function<PassingKind(size_t)> getPassingKind,
                     bool suppressSlashAfterSelf = false, char slash = '/',
                     StringRef plus = "+");
  PassingKindPrinter(raw_ostream &os, PogListAttr pogListAttr,
                     bool suppressSlashAfterSelf = false, char slash = '/',
                     StringRef plus = "+");
  PassingKindPrinter(AsmPrinter &printer, PogListAttr pogListAttr,
                     char slash = '/', StringRef plus = "+");

  /// Print a single '*' or '/' if needed, given the index of the passing kind.
  void printOptionalStarSlash(size_t idx);

  /// Print a single trailing '/' at the end of a signature if needed.
  void printOptionalTrailingSlash(size_t idx) const;

private:
  raw_ostream &os;
  size_t numPogs;
  std::function<PassingKind(size_t)> getPassingKind;
  PassingKind prevPassingKind;
  bool suppressSlashAfterSelf;
  char slash; // TODO: remove this when AsmParser can handle '/'.
  StringRef plus;
};

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
// Constraint Implication
//===----------------------------------------------------------------------===//

/// Check if propA logically implies propB.
/// Uses canonicalization, weakening rules (A implies A OR B),
/// conjunction elimination ((A AND B) implies A), and set-containment
/// subsumption for TypeConformsToTraitAttr (relies on conforms_to attrs
/// being canonicalized at construction to include ancestor traits).
/// Returns true if propA implies propB.
bool constraintImplies(TypedAttr propA, TypedAttr propB);

/// Check if propA and propB are logically contradictory.
/// Two propositions contradict if their conjunction is necessarily false.
/// E.g., X and NOT(X) contradict, as do (X AND Y) and NOT(X).
/// Uses canonicalization and recursive decomposition of AND/NOT expressions.
/// Returns true if propA and propB contradict.
bool constraintsContradict(TypedAttr propA, TypedAttr propB);

/// Result of checking whether a type conforms to a trait.
/// This is a 3-state result because conditional conformances may not be
/// provable at compile time.
enum class ConformanceResult {
  Yes,          // Definitely conforms (unconditional or constraint proven true)
  No,           // Definitely does not conform (constraint proven false)
  NeedsEvidence // Conditional conformance that can't be proven statically
};

/// Evaluate a conditional constraint using a pre-built evaluator that already
/// has struct parameters bound.
/// Shared by the parser's doesNominalTypeConformTo and CheckLifetimes'
/// destructor resolution.
ConformanceResult
evaluateConstraint(ParameterEvaluator &evaluator, ConstraintAttr constraint,
                   ArrayRef<ConstraintAttr> callerAssumptions = {});

/// TODO: `ClosureEmitter.cpp` has a nearly identical helper
/// (`getUnderlyingParamRef`). Unify these implementations to avoid drift.
///
/// Extract the underlying ParamDeclRefAttr from a type expression by peeling
/// UpcastAttr and TypeParamAttr/ParamType wrappers.
/// Returns a null attr if no ParamDeclRefAttr can be extracted.
ParamDeclRefAttr extractParamDeclRef(TypedAttr attr);

} // namespace LIT

} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
