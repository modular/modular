//===- MetaTypeConverter.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_METADIALECT_METATYPECONVERTER_H
#define KGEN_METADIALECT_METATYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

//===----------------------------------------------------------------------===//
// Meta to LLVM Type Converter
//===----------------------------------------------------------------------===//

/// This type converter maps fully-specified meta dialect parametric types and
/// built-in MLIR types to LLVM types.
class MetaToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  MetaToLLVMTypeConverter(mlir::Location loc,
                          const mlir::LowerToLLVMOptions &options);

  /// Report an error or conversion failure.
  /// TODO: TypeConverter needs an error reporting mechanism.
  mlir::InFlightDiagnostic emitError(llvm::StringRef msg) {
    return mlir::emitError(loc) << msg;
  }

private:
  /// A location used to report conversion failures.
  mlir::Location loc;
};

#endif // KGEN_METADIALECT_METATYPECONVERTER_H
