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

/// Mangle a parameter name with the line and column index where it's declared.
inline std::string mangleParameter(StringRef name, unsigned line,
                                   unsigned col) {
  return ("_" + Twine(line) + "x" + Twine(col) + "_" + name).str();
}

/// Recursively demangle the parameter names (declaration of references) in the
/// given mlir type or attribute, if necessary.
template <typename AttrOrType>
static AttrOrType demangleIfNeeded(AttrOrType arg) {
  auto demangle = [](auto declOrRef) {
    llvm::Regex re("^_[0-9]+x[0-9]+_");
    if (StringRef name = declOrRef.getName(); re.match(name))
      return decltype(declOrRef)::get(re.sub("", name),
                                      demangleIfNeeded(declOrRef.getType()));
    return declOrRef;
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
