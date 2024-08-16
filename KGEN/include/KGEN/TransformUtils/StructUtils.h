//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_STRUCTUTILS_H
#define KGEN_TRANSFORMUTILS_STRUCTUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/ValueRange.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::KGEN {
/// Given a type, if it is a `!kgen.struct`, recursively flatten its element
/// types and append them into `types`.
void flattenTypeIfStruct(Type type, SmallVectorImpl<Type> &types);

/// Given a value that might be a struct, recursively unpack it into its
/// flattened element types.
void flattenAndUnpackIfStruct(mlir::RewriterBase &b, Location loc, Value value,
                              SmallVectorImpl<Value> &values);

/// Given a type that might be a struct, recursively reconstruct a value of that
/// type from constituent elements pointed to by `it`.
Value flattenAndPackIfStruct(mlir::RewriterBase &b, Location loc, Type type,
                             ValueRange::iterator &it);

/// Flatten structs in block arguments, replacing uses of the old block
/// arguments with reconstructed structs from the new block arguments.
void flattenStructsInArguments(mlir::RewriterBase &b, Location loc,
                               Block *body);

/// Flatten structs in an operation's operand list, extracting the constituent
/// element values from the input structs.
void flattenStructsInOperands(mlir::RewriterBase &b, Operation *op);

/// Flatten structs in an operation's results, replacing the results with new
/// results comprised of the constituent element values. The original struct
/// values are reconstructed and used to replace the original op.
///
/// This will delete the operation passed in! The new operation is returned.
Operation *flattenStructsInResults(mlir::RewriterBase &b, Operation *op);
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_STRUCTUTILS_H
