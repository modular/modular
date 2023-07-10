//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Regex.h"

namespace M::KGEN {
class DeclRefType;
namespace LIT {
class StructFieldOp;
class NoneType;
class LifetimeType;

/// Mangle a parameter name with the line and column index where it's declared.
inline std::string mangleParameter(StringRef name, unsigned line,
                                   unsigned col) {
  return ("_" + Twine(line) + "x" + Twine(col) + "_" + name).str();
}

/// Demangle a mangled parameter name if it is mangled.
inline StringRef demangleParameterName(StringRef name) {
  llvm::Regex re("^_[0-9]+x[0-9]+_");
  if (!re.match(name))
    return name;
  // Strip the prefix. Drop the leading underscore and the drop until the second
  // underscore. This way, the function can avoid returning a `std::string`.
  name = name.drop_front();
  return name.drop_front(name.find('_') + 1);
}

/// Recursively demangle the parameter names (declaration of references) in the
/// given mlir type or attribute, if necessary.
template <typename AttrOrType>
AttrOrType demangleIfNeeded(AttrOrType arg) {
  auto demangle = [](auto declOrRef) {
    return decltype(declOrRef)::get(demangleParameterName(declOrRef.getName()),
                                    demangleIfNeeded(declOrRef.getType()));
  };

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](ParamDeclRefAttr declRef) { return demangle(declRef); });
  replacer.addReplacement([&](ParamDeclAttr decl) { return demangle(decl); });
  return cast<AttrOrType>(replacer.replace(arg));
}
} // namespace LIT
} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

#endif // KGEN_LITDIALECT_LITATTRS_H
