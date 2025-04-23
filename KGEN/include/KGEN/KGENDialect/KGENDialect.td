//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the KGENDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDIALECT_TD
#define KGEN_KGENDIALECT_KGENDIALECT_TD

include "mlir/IR/OpBase.td"

def KGENDialect : Dialect {
  let name = "kgen";

  let summary = "The 'kgen' dialect";
  let description = [{
    This dialect defines the `kgen` dialect, which defines the top level
    and other support operations for kernel generation framework.
  }];

  let cppNamespace = "::M::KGEN";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let dependentDialects = ["MDialect", "InterpreterDialect"];

  let extraClassDeclaration = [{
    /// Register all KGEN attributes.
    void registerAttributes();
    /// Register all KGEN types.
    void registerTypes();
    /// Inject any external attribute models.
    void injectAttrInterfaces();

    using TypeParseFn = Type (*)(AsmParser &);
    using TypePrintFn = void (*)(AsmPrinter &, Type);

    /// Register a keyword parser. A keyword can only be registered once.
    /// Conflicts are up to dialect users to manage. The only restriction on
    /// which keywords can be registered is that they cannot collide with
    /// builtin MLIR types: `i32`, `index`, etc, which are also identified with
    /// keywords.
    void registerKeywordParser(StringRef keyword, TypeParseFn parse);

    /// Register a keyword parser and printer. The dialect will try to parse
    /// a pretty type indicated by a keyword when parsing a parameter type.
    void registerPrettyType(StringRef keyword, TypeParseFn parse, mlir::TypeID id,
                            TypePrintFn print);

    /// Register a custom type parser using the type's mnemonic.
    template <typename TypeT>
    void registerMnemonicType();

    /// Get the "name" of the type, if it has registered one.
    std::optional<StringRef> getTypeName(mlir::TypeID id);

    /// Set the special parser for symbol references.
    void setSymbolTypeParser(
        std::function<FailureOr<Type>(AsmParser &, SymbolRefAttr)> parser) {
      symbolTypeParser = std::move(parser);
    }

  private:
    /// The map of keyword parsers.
    llvm::StringMap<TypeParseFn> typeParseFns;
    /// A map of pretty type printers.
    DenseMap<mlir::TypeID, TypePrintFn> typePrintFns;
    /// A map containing the "names" of types.
    DenseMap<mlir::TypeID, StringRef> typeNames;

    /// Function that parses symbol references as types.
    std::function<FailureOr<Type>(AsmParser &, SymbolRefAttr)> symbolTypeParser;

    /// Allow the type pretty printer and parser utilities to access the
    /// pretty functions.
    friend OptionalParseResult parseOptionalKGENType(AsmParser &, Type &);
    friend void printKGENType(AsmPrinter &, Type);
    friend void printParamName(AsmPrinter &p, StringAttr, bool);
  }];
}

#endif // KGEN_KGENDIALECT_KGENDIALECT_TD
