//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITTYPES_H
#define KGEN_LITDIALECT_LITTYPES_H

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "Support/ForwardDecls.h"

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

namespace M::KGEN::LIT {
class FnMetadataAttr;

class LITSignatureType : public SignatureType {
public:
  using SignatureType::SignatureType;
  LITSignatureType(SignatureType sig);

  /// Get the signature metadata.
  FnMetadataAttr getMetadata();

  /// Get the function input argument names.
  ArrayRef<StringAttr> getArgNames();

  /// Return the name for the specified value input argument.
  StringAttr getArgName(size_t inputNo);

  /// Get the function default arguments.
  ArrayRef<TypedAttr> getDefaultArguments();

  /// A `SignatureType` is a LIT signature if it contains function metadata.
  static bool classof(SignatureType type);
  static bool classof(Type type);

  static LITSignatureType get(MLIRContext *ctx, TypeRange inputs = {},
                              TypeRange results = {});
  static LITSignatureType get(FunctionType values,
                              TypeArrayAttr inputParams = {},
                              TypeArrayAttr resultParams = {},
                              ArrayRef<ValueInputConvention> convs = {},
                              FnEffects effects = {}, Attribute metadata = {});
};

/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type);
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.h.inc"

#endif // KGEN_LITDIALECT_LITTYPES_H
