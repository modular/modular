//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EmitFuncHeader.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/ML/DType.h"
#include "Support/STLExtras.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

/// Emit the C signature of a KGEN func.
static LogicalResult emitSignature(raw_ostream &os, SymbolTable &symtab,
                                   FuncOp func) {
  auto printDTypeAsC = [&](DType dt) -> LogicalResult {
    if (dt.isFloat()) {
      switch (dt.getValue()) {
      case DType::f32:
        os << "float";
        return success();
      case DType::f64:
        os << "double";
        return success();
      }
      return func.emitError("unhandled floating point dtype: ")
             << dt.getAsString();
    }
    if (dt.isInt()) {
      os << (dt.isUInt() ? "u" : "") << "int" << dt.getWidthInBits() << "_t";
      return success();
    }
    if (dt == DType::invalid) {
      os << "void";
      return success();
    }
    if (dt.isBool()) {
      os << "bool";
      return success();
    }

    return func.emitError("unhandled dtype for header generation ")
           << dt.getAsString();
  };

  // Helper to print a function as a C type.
  std::function<LogicalResult(Type)> printTypeAsC =
      [&](Type t) -> LogicalResult {
    if (auto simd = dyn_cast<POP::SIMDType>(t)) {
      // Since the vector_size attribute only works on GNU and CLANG compilers,
      // we pass in an array.
      if (failed(printDTypeAsC(*simd.getResolvedDType())))
        return failure();
      auto size = *simd.getResolvedSize();
      // size == 1 is a scalar
      if (size != 1)
        os << "[" << size << "]";
      return success();
    }

    if (auto ptr = dyn_cast<POP::PointerType>(t)) {
      if (Type type = ptr.getResolvedElementType()) {
        if (failed(printTypeAsC(type)))
          return failure();
      } else {
        os << "void";
      }
      os << " *";
      return success();
    }

    if (auto array = dyn_cast<POP::ArrayType>(t)) {
      if (failed(printTypeAsC(array.getResolvedElementType())))
        return failure();
      os << "[" << *array.getResolvedSize() << "]";
      return success();
    }

    if (auto buffer = dyn_cast<ZAP::BufferType>(t)) {
      os << "void *, ssize_t, uint8_t";
      return success();
    }
    if (auto ndbuffer = dyn_cast<ZAP::NDBufferType>(t)) {
      os << "void *, ssize_t, ssize_t[5], uint8_t";
      return success();
    }

    if (auto structType = dyn_cast<POP::StructType>(t)) {
      SmallVector<Type> elementTypes;
      elementTypes.reserve(structType.getElementTypes().size());
      if (failed(structType.resolveElementTypes(elementTypes)))
        return failure();
      for (auto &elTy : llvm::enumerate(elementTypes)) {
        if (elTy.index() != 0)
          os << ", ";
        if (failed(printTypeAsC(elTy.value())))
          return failure();
      }
      return success();
    }

    if (t.isa<DTypeType>()) {
      os << "uint8_t";
      return success();
    }

    if (auto ref = dyn_cast<RefType>(t)) {
      auto decl = symtab.lookup<StructDeclOp>(ref.getName().getAttr());
      ParameterEvaluator evaluator;
      for (ParamBindAttr bind : ref.getParamValues())
        evaluator.setParameterValue(bind.getDecl(), bind.getValue());
      assert(decl && "expected a valid type reference");
      return failableInterleave(
          decl.getFieldDecls(),
          [&](StructFieldOp field) {
            return printTypeAsC(evaluator.getReboundType(field.getType()));
          },
          [&] {
            os << ", ";
            return mlir::success();
          });
    }

    if (!t.isa<IndexType, IntegerType, FloatType>())
      return func.emitError("unsupported argument type: ") << t;
    if (!t.isIndex() && !llvm::isPowerOf2_64(t.getIntOrFloatBitWidth()))
      return func.emitError("integer or float bitwidth must be a power of 2");

    // Elementary type, just print it.
    if (t.isa<IntegerType>())
      os << "int" << t.getIntOrFloatBitWidth() << "_t";
    else if (t.isa<IndexType>())
      os << "ssize_t";
    else if (t.isF16())
      llvm::report_fatal_error("no support for fp16 yet");
    else if (t.isF32())
      os << "float";
    else if (t.isF64())
      os << "double";
    else
      return func.emitError("unhandled float type: ") << t;
    return success();
  };

  // Now print the function declaration.
  os << "extern ";
  if (func.getNumResults() > 1)
    return func.emitError("functions with more than 1 result unsupported");
  if (func.getNumResults() == 0)
    os << "void";
  else if (failed(printTypeAsC(func.getResultTypes().front())))
    return failure();
  os << " " << func.getName() << "(";
  for (auto &it : llvm::enumerate(func.getFunctionType().getInputs())) {
    if (it.index() != 0)
      os << ", ";
    if (failed(printTypeAsC(it.value())))
      return failure();
  }
  os << ");";
  return success();
}

/// This allows us to emit a header file for the given func so that we can
/// `#include` it and get nice autocompletion/etc. in users' IDEs.
LogicalResult M::KGEN::emitHeaderForFunc(SymbolTable &symtab, FuncOp func,
                                         StringRef filename) {
  std::string err;
  auto outFile = mlir::openOutputFile(filename, &err);
  if (!outFile)
    return mlir::emitError(func.getLoc(), err);

  llvm::StringLiteral headerFmtStart = R"literal(//===-{0}-===//
//
// This file is Modular Inc proprietary.
//
//==={1}===//
// THIS FILE IS AUTOGENERATED BY `kgen`, DO NOT EDIT!
//==={1}===//

#ifndef {2}
#define {2}

#ifdef __cplusplus
extern "C" {{
#endif

#include <stdint.h>
#include <stddef.h>

)literal";

  llvm::StringLiteral headerFmtEnd = R"literal(
#ifdef __cplusplus
} // extern "C"
#endif

#endif // {0}
)literal";

  std::string headerGuard = "__KGEN_" + func.getName().upper() + "_H";
  outFile->os() << llvm::formatv(
      headerFmtStart.data(),
      llvm::fmt_align(" " + func.getName() + ".h ", llvm::AlignStyle::Left,
                      80 - 2 * strlen("//===-"), '-'),
      llvm::fmt_repeat('-', 80 - 2 * strlen("//===")), headerGuard);
  if (failed(emitSignature(outFile->os(), symtab, func)))
    return mlir::emitError(func.getLoc(),
                           "during header emission for this function");
  outFile->os() << llvm::formatv(headerFmtEnd.data(), headerGuard);

  outFile->keep();

  return mlir::success();
}
