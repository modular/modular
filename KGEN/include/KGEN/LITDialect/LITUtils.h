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
class ParamDeclAttr;
class ParamDeclArrayAttr;
class ParameterEvaluator;
class ParameterExprArrayAttr;

namespace LIT {
class PassingKindArrayAttr;
enum class PassingKind : uint32_t;

/// Returns whether the given attribute is a LIT type expression.
bool isTypeExpr(TypedAttr attr);

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

/// Mangle a parameter name with the line and column index where it's declared.
std::string mangleParameter(const Twine &baseName, unsigned line, unsigned col);

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

/// Pretty print a nested symbol reference to a name.
void printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol);

/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present. If `hasAddress` is set, the
/// default value is parsed as if `type` is an address type: either a pointer or
/// reference. The method is tolerant if `type` is not actually one.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type, bool hasAddress = false);

/// Parse and print a ParamDeclAttr which has syntactic form `declName ([ name
/// ])? (: declType )?`. `name` is the unmangled name (i.e. as the user declared
/// it).
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result,
                           StringAttr &name);
void printParamDecl(AsmPrinter &p, ParamDeclAttr decl, StringAttr name);

/// Parse a parameter specification in a lit op.
ParseResult
parseOptionalParameterSpec(AsmParser &p, ParamDeclArrayAttr &inputParamDecls,
                           ParamDeclArrayAttr &resultParamDecls,
                           SmallVectorImpl<StringAttr> &paramNames,
                           SmallVectorImpl<PassingKind> &paramPassingKinds,
                           SmallVectorImpl<TypedAttr> &defaultPosParams);

/// Print a parameter specification in a lit op. A ParameterEvaluator is
/// necessary to substitute parameters into parametric parameters.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> inputParamDecls,
                                ArrayRef<ParamDeclAttr> resultParamDecls,
                                ArrayRef<StringAttr> paramNames,
                                ArrayRef<PassingKind> paramPassingKinds,
                                ArrayRef<TypedAttr> defaultPosParams,
                                ParameterEvaluator &evaluator);

/// Parse a parameter signature (input/result types with optional default
/// values) if present.
ParseResult
parseOptionalParamSignature(AsmParser &p,
                            SmallVectorImpl<Type> &inputParamTypes,
                            SmallVectorImpl<Type> &resultParamTypes,
                            SmallVectorImpl<StringAttr> &paramNames,
                            SmallVectorImpl<PassingKind> &paramPassingKinds,
                            SmallVectorImpl<TypedAttr> &defaultPosParams);

/// Print the parameter type signature if there are any input or result types,
/// along with the default input parameter values.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 ArrayRef<Type> resultParamTypes,
                                 ArrayRef<StringAttr> paramNames,
                                 ArrayRef<PassingKind> paramPassingKinds,
                                 ArrayRef<TypedAttr> defaultPosParams);

/// Parse an optional parameter or argument name.
ParseResult parseOptionalName(AsmParser &p, StringAttr &name);

/// Count the number of positional-only passing kinds.
size_t countNumPosOnly(ArrayRef<PassingKind> kinds);

/// Count the number of implicit passing kinds.
size_t countNumImplicitKinds(ArrayRef<PassingKind> kinds);

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

/// Handles parsing '|' and '*' in lit IR and counts the number of arguments of
/// different passing kinds.
/// TODO(#23387): fix this when AsmParser can handle '/'.
class PassingKindParser {
public:
  PassingKindParser(AsmParser &parser) : parser(parser) {}

  /// Try to parse a single optional '*' or '|', and emit an error if a
  /// duplicate is found or a '|' comes after a '*'.
  OptionalParseResult parseOptionalStarSlash();

  /// Populate the parameter passing kinds.
  void populatePassingKinds(SmallVectorImpl<PassingKind> &kinds) const;

  /// Return true if an implicit parameter was seen. I.e., the parser is
  /// currently parsing an implicit parameter.
  bool isCurrentImplicit() const { return foundImplicit; }

private:
  /// Return the number of positional-only, positional-or-keyword, keyword-only
  /// and implicit arguments seen so far, respectively.
  std::tuple<size_t, size_t, size_t, size_t> getNumPassingKinds() const;

  AsmParser &parser;
  size_t idx = 0;
  size_t numPosOnly = 0;
  size_t numPosOrKw = 0;
  size_t numKwOnly = 0;
  bool foundSlash = false;
  bool foundStar = false;
  bool foundImplicit = false;
};

/// Handles printing '/' and '*' in lit IR. Optionally, it allows specifying a
/// character to be used instead of '/'. It also allows specifying a flag to
/// suppress the '/' if it immediately follows the first argument (useful if
/// printing methods with mojo syntax).
class PassingKindPrinter {
public:
  PassingKindPrinter(raw_ostream &os, size_t numInputs,
                     bool suppressSlashAfterSelf = false, char slash = '/');
  PassingKindPrinter(AsmPrinter &printer, size_t numInputs, char slash = '/');

  /// Print a single '*' or '/' if needed, given the passing kind, and the index
  /// of the argument.
  void printOptionalStarSlash(PassingKind passingKind, size_t idx);

  /// Print a single trailing '/' at the end of a signature if needed.
  void printOptionalTrailingSlash(size_t idx) const;

private:
  raw_ostream &os;
  size_t numInputs;
  PassingKind prevPassingKind;
  bool suppressSlashAfterSelf;
  char slash; // TODO: remove this when AsmParser can handle '/'.
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

} // namespace LIT
} // namespace KGEN
} // namespace M

#endif // KGEN_LITDIALECT_LITUTILS_H
