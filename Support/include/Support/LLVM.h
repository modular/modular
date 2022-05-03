//===- Support/LLVM.h -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM and MLIR datatypes
// that we want to use unqualified.
//
// Note that most of these are forward declared and then imported into the
// M (Modular) namespace with using decls, rather than being #included.  This is
// because we want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LLVM_H
#define SUPPORT_LLVM_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Can not forward declare inline functions with default arguments, so we
// include the header directly.
#include "mlir/Support/LogicalResult.h"

// Forward declarations of LLVM classes to be imported in to the M (Modular)
// namespace.
namespace llvm {
template <typename KeyT, typename ValueT, unsigned InlineBuckets,
          typename KeyInfoT, typename BucketT>
class SmallDenseMap;
} // namespace llvm

// Import classes from the `mlir` namespace into the `M` namespace.  All
// of the following classes have been already forward declared and imported from
// `llvm` in to the `mlir` namespace. For classes with default template
// arguments, MLIR does not import the type directly, it creates a templated
// using statement. This is due to the limitiation that only one declaration of
// a type can have default arguments. For those types, it is important to import
// the MLIR version, and not the LLVM version. To keep things simple, all
// classes here should be imported from the `mlir` namespace, not the `llvm`
// namespace.
namespace M {
using llvm::SmallDenseMap;
using mlir::APFloat;
using mlir::APInt;
using mlir::APSInt;
using mlir::ArrayRef;
using mlir::cast;
using mlir::cast_or_null;
using mlir::DenseMap;
using mlir::DenseMapInfo;
using mlir::DenseSet;
using mlir::dyn_cast;
using mlir::dyn_cast_or_null;
using mlir::function_ref;
using mlir::isa;
using mlir::isa_and_nonnull;
using mlir::iterator_range;
using mlir::MutableArrayRef;
using mlir::None;
using mlir::Optional;
using mlir::PointerUnion;
using mlir::raw_ostream;
using mlir::SmallPtrSet;
using mlir::SmallPtrSetImpl;
using mlir::SmallString;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::StringLiteral;
using mlir::StringRef;
using mlir::StringSet;
using mlir::TinyPtrVector;
using mlir::Twine;
using mlir::TypeSwitch;
} // namespace M

// Forward declarations of classes to be imported in to the M namespace.
namespace mlir {
class ArrayAttr;
class AsmParser;
class AsmPrinter;
class Attribute;
class BFloat16Type;
class Block;
class BlockAndValueMapping;
class BlockArgument;
class BoolAttr;
class Builder;
class NamedAttrList;
class ConversionPattern;
class ConversionPatternRewriter;
class ConversionTarget;
class DenseElementsAttr;
class DenseIntElementsAttr;
class Diagnostic;
class Dialect;
class DialectAsmParser;
class DialectAsmPrinter;
class DialectRegistry;
class DictionaryAttr;
class ElementsAttr;
class FileLineColLoc;
class FlatSymbolRefAttr;
class FloatAttr;
class FloatType;
class FunctionType;
class FusedLoc;
class ImplicitLocOpBuilder;
class IndexType;
class InFlightDiagnostic;
class IntegerAttr;
class IntegerType;
class Location;
class MemRefType;
class MLIRContext;
class ModuleOp;
class MutableOperandRange;
class NamedAttribute;
class NamedAttrList;
class NoneType;
class OpAsmDialectInterface;
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class OperandRange;
class Operation;
class OpFoldResult;
class OpOperand;
class OpResult;
class OwningModuleRef;
class ParseResult;
class Pass;
class PatternRewriter;
class RankedTensorType;
class Region;
class RewritePatternSet;
class ShapedType;
class SplatElementsAttr;
class StringAttr;
class SymbolRefAttr;
class SymbolTable;
class SymbolTableCollection;
class TupleType;
class Type;
class TypeAttr;
class TypeConverter;
class TypeID;
class TypeRange;
class TypeStorage;
class UnknownLoc;
class Value;
class ValueRange;
class VectorType;
class WalkResult;
enum class RegionKind;
struct CallInterfaceCallable;
struct LogicalResult;
struct MemRefAccess;
struct OperationState;

template <typename T>
class FailureOr;

template <typename SourceOp>
class OpConversionPattern;
template <typename T>
class OperationPass;
template <typename SourceOp>
struct OpRewritePattern;
template <typename OpTy>
class OwningOpRef;

using DefaultTypeStorage = TypeStorage;
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;
namespace OpTrait {}
namespace quant {
class QuantizedType;
class UniformQuantizedType;
class UniformQuantizedPerAxisType;
} // namespace quant
} // namespace mlir

// Import things we want into our namespace.
namespace M {
// clang-tidy removes following using directives incorrectly. So force
// clang-tidy to ignore them.
// TODO: It is better to use `NOLINTBEGIN/END` comments to disable clang-tidy
// than adding `NOLINT` to every line. `NOLINTBEGIN/END` will supported from
// clang-tidy-14.
using mlir::ArrayAttr;                 // NOLINT(misc-unused-using-decls)
using mlir::AsmParser;                 // NOLINT(misc-unused-using-decls)
using mlir::AsmPrinter;                // NOLINT(misc-unused-using-decls)
using mlir::Attribute;                 // NOLINT(misc-unused-using-decls)
using mlir::BFloat16Type;              // NOLINT(misc-unused-using-decls)
using mlir::Block;                     // NOLINT(misc-unused-using-decls)
using mlir::BlockAndValueMapping;      // NOLINT(misc-unused-using-decls)
using mlir::BlockArgument;             // NOLINT(misc-unused-using-decls)
using mlir::BoolAttr;                  // NOLINT(misc-unused-using-decls)
using mlir::Builder;                   // NOLINT(misc-unused-using-decls)
using mlir::CallInterfaceCallable;     // NOLINT(misc-unused-using-decls)
using mlir::ConversionPattern;         // NOLINT(misc-unused-using-decls)
using mlir::ConversionPatternRewriter; // NOLINT(misc-unused-using-decls)
using mlir::ConversionTarget;          // NOLINT(misc-unused-using-decls)
using mlir::DefaultTypeStorage;        // NOLINT(misc-unused-using-decls)
using mlir::DenseElementsAttr;         // NOLINT(misc-unused-using-decls)
using mlir::DenseIntElementsAttr;      // NOLINT(misc-unused-using-decls)
using mlir::Diagnostic;                // NOLINT(misc-unused-using-decls)
using mlir::Dialect;                   // NOLINT(misc-unused-using-decls)
using mlir::DialectAsmParser;          // NOLINT(misc-unused-using-decls)
using mlir::DialectAsmPrinter;         // NOLINT(misc-unused-using-decls)
using mlir::DialectRegistry;           // NOLINT(misc-unused-using-decls)
using mlir::DictionaryAttr;            // NOLINT(misc-unused-using-decls)
using mlir::ElementsAttr;              // NOLINT(misc-unused-using-decls)
using mlir::failed;                    // NOLINT(misc-unused-using-decls)
using mlir::failure;                   // NOLINT(misc-unused-using-decls)
using mlir::FailureOr;                 // NOLINT(misc-unused-using-decls)
using mlir::FileLineColLoc;            // NOLINT(misc-unused-using-decls)
using mlir::FlatSymbolRefAttr;         // NOLINT(misc-unused-using-decls)
using mlir::FloatAttr;                 // NOLINT(misc-unused-using-decls)
using mlir::FloatType;                 // NOLINT(misc-unused-using-decls)
using mlir::FunctionType;              // NOLINT(misc-unused-using-decls)
using mlir::FusedLoc;                  // NOLINT(misc-unused-using-decls)
using mlir::ImplicitLocOpBuilder;      // NOLINT(misc-unused-using-decls)
using mlir::IndexType;                 // NOLINT(misc-unused-using-decls)
using mlir::InFlightDiagnostic;        // NOLINT(misc-unused-using-decls)
using mlir::IntegerAttr;               // NOLINT(misc-unused-using-decls)
using mlir::IntegerType;               // NOLINT(misc-unused-using-decls)
using mlir::Location;                  // NOLINT(misc-unused-using-decls)
using mlir::LogicalResult;             // NOLINT(misc-unused-using-decls)
using mlir::MemRefAccess;              // NOLINT(misc-unused-using-decls)
using mlir::MemRefType;                // NOLINT(misc-unused-using-decls)
using mlir::MLIRContext;               // NOLINT(misc-unused-using-decls)
using mlir::ModuleOp;                  // NOLINT(misc-unused-using-decls)
using mlir::MutableOperandRange;       // NOLINT(misc-unused-using-decls)
using mlir::NamedAttribute;            // NOLINT(misc-unused-using-decls)
using mlir::NamedAttrList;             // NOLINT(misc-unused-using-decls)
using mlir::NoneType;                  // NOLINT(misc-unused-using-decls)
using mlir::OpAsmDialectInterface;     // NOLINT(misc-unused-using-decls)
using mlir::OpAsmParser;               // NOLINT(misc-unused-using-decls)
using mlir::OpAsmPrinter;              // NOLINT(misc-unused-using-decls)
using mlir::OpAsmSetValueNameFn;       // NOLINT(misc-unused-using-decls)
using mlir::OpBuilder;                 // NOLINT(misc-unused-using-decls)
using mlir::OpConversionPattern;       // NOLINT(misc-unused-using-decls)
using mlir::OperandRange;              // NOLINT(misc-unused-using-decls)
using mlir::Operation;                 // NOLINT(misc-unused-using-decls)
using mlir::OperationPass;             // NOLINT(misc-unused-using-decls)
using mlir::OperationState;            // NOLINT(misc-unused-using-decls)
using mlir::OpFoldResult;              // NOLINT(misc-unused-using-decls)
using mlir::OpOperand;                 // NOLINT(misc-unused-using-decls)
using mlir::OpResult;                  // NOLINT(misc-unused-using-decls)
using mlir::OpRewritePattern;          // NOLINT(misc-unused-using-decls)
using mlir::OwningModuleRef;           // NOLINT(misc-unused-using-decls)
using mlir::OwningOpRef;               // NOLINT(misc-unused-using-decls)
using mlir::ParseResult;               // NOLINT(misc-unused-using-decls)
using mlir::Pass;                      // NOLINT(misc-unused-using-decls)
using mlir::PatternRewriter;           // NOLINT(misc-unused-using-decls)
using mlir::RankedTensorType;          // NOLINT(misc-unused-using-decls)
using mlir::Region;                    // NOLINT(misc-unused-using-decls)
using mlir::RegionKind;                // NOLINT(misc-unused-using-decls)
using mlir::RewritePatternSet;         // NOLINT(misc-unused-using-decls)
using mlir::ShapedType;                // NOLINT(misc-unused-using-decls)
using mlir::SplatElementsAttr;         // NOLINT(misc-unused-using-decls)
using mlir::StringAttr;                // NOLINT(misc-unused-using-decls)
using mlir::succeeded;                 // NOLINT(misc-unused-using-decls)
using mlir::success;                   // NOLINT(misc-unused-using-decls)
using mlir::SymbolRefAttr;             // NOLINT(misc-unused-using-decls)
using mlir::SymbolTable;               // NOLINT(misc-unused-using-decls)
using mlir::SymbolTableCollection;     // NOLINT(misc-unused-using-decls)
using mlir::TupleType;                 // NOLINT(misc-unused-using-decls)
using mlir::Type;                      // NOLINT(misc-unused-using-decls)
using mlir::TypeAttr;                  // NOLINT(misc-unused-using-decls)
using mlir::TypeConverter;             // NOLINT(misc-unused-using-decls)
using mlir::TypeID;                    // NOLINT(misc-unused-using-decls)
using mlir::TypeRange;                 // NOLINT(misc-unused-using-decls)
using mlir::TypeStorage;               // NOLINT(misc-unused-using-decls)
using mlir::UnknownLoc;                // NOLINT(misc-unused-using-decls)
using mlir::Value;                     // NOLINT(misc-unused-using-decls)
using mlir::ValueRange;                // NOLINT(misc-unused-using-decls)
using mlir::VectorType;                // NOLINT(misc-unused-using-decls)
using mlir::WalkResult;                // NOLINT(misc-unused-using-decls)
using mlir::quant::QuantizedType;      // NOLINT(misc-unused-using-decls)
using mlir::quant::
    UniformQuantizedPerAxisType;         // NOLINT(misc-unused-using-decls)
using mlir::quant::UniformQuantizedType; // NOLINT(misc-unused-using-decls)

namespace OpTrait = mlir::OpTrait;
} // namespace M

#endif // SUPPORT_LLVM_H
