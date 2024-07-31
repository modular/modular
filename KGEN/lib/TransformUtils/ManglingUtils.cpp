//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/ManglingUtils.h"

#include "KGEN/KGENDialect/KGENUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// Return a string that is a unique specification of the specified parameter.
static void printParameterMangling(TypedAttr value, raw_ostream &os) {
  // TODO: Don't print #lit.lifetime, they are singletons!
  // if (isa<LifetimeAttr>(value))
  //  return;

  // It is very common to pass types in as parameters, since this is how
  // parameters of Trait type get substituted in.  We'd really like to include
  // just the full name of the type (e.g. something like
  // stdlib::builtin::string_literal::StringLiteral) instead of the full string
  // form of the TypeConstantAttr - which includes a vtable!

  // TODO(MOCO-945): we should have this mangled in in a stable way in the
  // vtable itself.  Until then, we do a "terrible hack" to get it because
  // everything will have a del member, and thus have something like:
  //   @"stdlib::builtin::string_literal::StringLiteral::__del__(
  //       stdlib::builtin::string_literal::StringLiteral)
  //  ... in the stringified form of the parameter.
  if (auto typeCst = dyn_cast<TypeConstantAttr>(value)) {
    VTableAttr vtable = typeCst.getVTable();

    // We need to mangle in the metatype of the TypeConstantAttr into the name
    // of the type, not just the type itself.  Consider something like:
    //   struct Thing[element_trait: _AnyTypeMetaType, type: element_trait]: ...
    // we need Thing[AnyType, Int] and Thing[Printable, Int] to mangle
    // differently.  We also need to mangle different named traits with the same
    // bodies, e.g. consider "trait MyAnyType: pass" and "Thing[AnyType, Int]"
    // vs "Thing[MyAnyType, Int]".
    //
    // Unfortunately, we currently lose the metatype completely because of
    // lower-lit, so we mangle in the list of requirements as a "incorrect but
    // close" way to unique them.
    // TODO(MOCO-945): We should have the metatype name directly in the vtable.
    std::string requirementsList;

    // Find the __del__ member if present.
    SymbolConstantAttr delMember;
    for (VTableEntryAttr entry : vtable.getEntries()) {
      if (entry.getName() == "__del__") {
        delMember = dyn_cast<SymbolConstantAttr>(entry.getMethod());
      } else {
        // Build a list of the non-del requirements as a gross approximation
        // for the metatype of the trait.
        if (!requirementsList.empty())
          requirementsList += ",";
        requirementsList += entry.getName().strref();
      }
    }

    // If we have a __del__ member, process it.
    if (delMember) {
      auto delMangledName = delMember.getSymbol().getLeafReference().strref();

      // Drop the stuff up to the argument name in the parens.
      size_t pos = delMangledName.find('(');
      assert(pos != StringRef::npos && "Mojo has predictable symbol mangling");
      delMangledName = delMangledName.drop_front(pos + 1);

      // The REPL will cons up names like:
      // "...__del__(Expression [6] wrapper::Sphere)_thunk", and we don't want
      // to treat this as something named Expression, so drop this.
      if (delMangledName.starts_with("Expression [")) {
        delMangledName = delMangledName.drop_front(strlen("Expression ["));
        pos = delMangledName.find(']');
        assert(pos != StringRef::npos && delMangledName[pos + 1] == ' ' &&
               "Mojo has predictable symbol mangling");
        delMangledName = delMangledName.drop_front(pos + 2);
      }

      pos = delMangledName.find(')');
      assert(pos != StringRef::npos && "Mojo has predictable symbol mangling");
      delMangledName = delMangledName.take_front(pos);

      // If the type had parameters, it will have them erased to something
      // like 'simd::SIMD[$0, $1]'.  These are pointless, so drop them.
      pos = delMangledName.find('[');
      if (pos != StringRef::npos)
        delMangledName = delMangledName.take_front(pos);

      assert(!delMangledName.empty() && "Should have something!");
      os << delMangledName;

      // Encode any parameter values.
      if (!delMember.getParamValues().empty()) {
        os << '[';
        llvm::interleaveComma(delMember.getParamValues(), os,
                              [&](TypedAttr paramValue) {
                                printParameterMangling(paramValue, os);
                              });
        os << ']';
      }

      // Include the requirement list (if not AnyType) to include a semblance
      // of metatype into this.
      // TODO(MOCO-945): Use the actual named metatype.
      if (!requirementsList.empty())
        os << '.' << requirementsList;
      return;
    }
    // TODO: We're also getting TypeConstantAttrs for signature types, like:
    //    (!kgen.pointer<none>, !kgen.pointer<none>, !kgen.pointer<none>, index,
    //       index, index) capturing -> !kgen.none
    // Which could surely be compressed somehow as well.
  }

  // Handle VariadicAttr of types as well, which are common for variadic packs.
  if (auto variadic = dyn_cast<VariadicAttr>(value)) {
    os << '[';
    llvm::interleaveComma(variadic.getValues(), os, [&](TypedAttr paramValue) {
      printParameterMangling(paramValue, os);
    });
    os << ']';
    return;
  }

  // The kgen representation will always be a valid choice.
  os << getParamAsString(value);
}

std::string KGEN::mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<TypedAttr> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return generator.getName().str();

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  // Mangle in things like "size=42" for each of the parameters to make it easy
  // to read the resultant symbol and also to make things unique when
  // instantiated with different values.
  auto inputParamDecls = generator.getInputParamsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterMangling(value, os);
  }

  // Having "@" in mangled names is invalid for ELF files and triggers error at
  // linking stage, so replace them.
  std::replace(result.begin(), result.end(), '@', '_');
  return result;
}
