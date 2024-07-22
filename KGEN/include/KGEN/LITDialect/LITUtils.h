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

#include "KGEN/LITDialect/LITAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
class SymbolOpInterface;
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
class LITSignatureType;
enum class PassingKind : uint32_t;

/// Returns whether the given attribute is a LIT type expression.
bool isTypeExpr(TypedAttr attr);

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

/// Demangle a mangled parameter name if it is mangled.
StringRef demangleParameterName(StringRef name);

namespace impl {
Attribute demangleIfNeeded(Attribute arg);
Type demangleIfNeeded(Type arg);
} // namespace impl

/// Recursively demangle the parameter names (declaration of references) in the
/// given mlir type or attribute, if necessary.
template <typename AttrOrType>
AttrOrType demangleIfNeeded(AttrOrType arg) {
  return cast<AttrOrType>(impl::demangleIfNeeded(arg));
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print/Parse a (potentially) parametric mutability specifier and then a
/// value.  The three forms are: "imm expr", "mut expr", "mut=<expr>, expr"
/// without quotes.
ParseResult parseLifetimeParamValue(AsmParser &p, TypedAttr &result);
void printLifetimeParamValue(AsmPrinter &p, TypedAttr value);
inline void printLifetimeParamValue(AsmPrinter &p, Operation *,
                                    TypedAttr value) {
  printLifetimeParamValue(p, value);
}

/// Pretty print a nested symbol reference to a name.
void printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol);

/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present. If `hasAddress` is set, the
/// default value is parsed as if `type` is an address type: either a pointer or
/// reference. The method is tolerant if `type` is not actually one.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type, bool hasAddress = false);

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

/// Parse a parameter signature (input/result types with optional default
/// values) if present.
ParseResult parseOptionalParamSignature(AsmParser &p,
                                        SmallVectorImpl<Type> &inputParamTypes,
                                        PogListAttr &paramListAttr);

/// Print the parameter type signature if there are any input or result types,
/// along with the default input parameter values.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 PogListAttr paramListAttr);

/// Parse an optional parameter or argument name.
ParseResult parseOptionalName(AsmParser &p, StringAttr &name);

/// Parse an optional passing convention and variadicness. The the given index
/// will be added to the appropriate index array if a variadicness is present.
ParseResult parseConventionAndVariadicness(
    AsmParser &p, ArgConvention &convention,
    SmallVectorImpl<size_t> &variadicIndices, ssize_t &argPackIndex,
    std::optional<ArgConvention> &origArgPackConvention, size_t idx);

enum class Variadicness : uint8_t;
/// Print an optional passing convention and variadicness.
void printConventionAndVariadicness(AsmPrinter &p, ArgConvention convention,
                                    Variadicness variadicness);

/// Parse and print a lifetime set.
ParseResult parseLifetimeSet(AsmParser &p,
                             SmallVectorImpl<TypedAttr> &lifetimes);
OptionalParseResult
parseOptionalLifetimeSet(AsmParser &p, SmallVectorImpl<TypedAttr> &lifetimes);
void printLifetimeSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes);

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

/// Helper enum to make printing of variadicness easier.
enum class Variadicness : uint8_t { kNone, kVariadic, kPack };

/// Return an array of enums representing the variadicness of each
/// argument/parameter in the given list.
SmallVector<Variadicness> getVariadicness(PogListAttr pogListAttr);

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
// DefaultValueHandler
//===----------------------------------------------------------------------===//

/// Helper class to allow easy checking and retrieval of positional and
/// keyword-only default values.
class DefaultValueHandler {
public:
  DefaultValueHandler(ArrayRef<PogMetadataAttr> pogs,
                      ArrayRef<TypedAttr> defaultsPos,
                      ArrayRef<TypedAttr> defaultsKwOnly)
      : pogs(pogs), defaultsPos(defaultsPos), defaultsKwOnly(defaultsKwOnly),
        numPositional(countNumInferredKinds(pogs) + countNumPositional(pogs)),
        defaultPosStart(numPositional - defaultsPos.size()),
        kwOnlyEnd(pogs.size() - countNumImplicitKinds(pogs)),
        defaultKwOnlyStart(kwOnlyEnd - defaultsKwOnly.size()) {}

  DefaultValueHandler(PogListAttr pogListAttr);

  /// If the given index refers to an optional positional (pos-only or
  /// pos-or-kw) argument/parameter, return its default value or null otherwise.
  inline TypedAttr getPosDefault(size_t idx) {
    if (defaultPosStart <= idx && idx < numPositional)
      return defaultsPos[idx - defaultPosStart];
    return {};
  }

  /// If the given index refers to an optional keyword-only argument/parameter,
  /// return its default value or null otherwise.
  inline TypedAttr getKwOnlyDefault(size_t idx) {
    if (defaultKwOnlyStart <= idx && idx < kwOnlyEnd)
      return defaultsKwOnly[idx - defaultKwOnlyStart];
    return {};
  }

  /// If the given index refers to an optional argument/parameter (of any
  /// passing kind), return its default value or null otherwise.
  inline TypedAttr getDefault(size_t idx) {
    if (TypedAttr defaultOr = getPosDefault(idx))
      return defaultOr;
    return getKwOnlyDefault(idx);
  }

private:
  ArrayRef<PogMetadataAttr> pogs;
  ArrayRef<TypedAttr> defaultsPos;
  ArrayRef<TypedAttr> defaultsKwOnly;
  size_t numPositional;
  size_t defaultPosStart;
  size_t kwOnlyEnd;
  size_t defaultKwOnlyStart;
};

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

/// Verify the types of parameter or argument defaults, taking into account
/// their input conventions if applicable. This assumes that the passing kinds
/// and the number of defaults are valid.
LogicalResult verifyDefaultTypes(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<TypedAttr> defaultsPos,
                                 ArrayRef<TypedAttr> defaultsKwOnly,
                                 PogListAttr pogListAttr, ArrayRef<Type> types,
                                 StringRef argOrParam,
                                 ArrayRef<ArgConvention> convs = {});

/// Verify the the order of passing kinds, and that the number of defaults
/// doesn't exceed the number of corresponding passing kinds.
LogicalResult verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<PogMetadataAttr> pogs,
                                 size_t numPosDefaults,
                                 size_t numKwOnlyDefaults,
                                 StringRef argOrParam);

} // namespace LIT
} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
