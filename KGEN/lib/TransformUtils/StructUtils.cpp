//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/StructUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;

void KGEN::flattenTypeIfStruct(Type type, SmallVectorImpl<Type> &types) {
  if (auto structType = dyn_cast<StructType>(type)) {
    for (Type type : structType.getElementTypes())
      flattenTypeIfStruct(type, types);
    return;
  }
  types.push_back(type);
}

void KGEN::flattenAndUnpackIfStruct(mlir::RewriterBase &b, Location loc,
                                    Value value,
                                    SmallVectorImpl<Value> &values) {
  if (auto structType = dyn_cast<StructType>(value.getType())) {
    for (auto [i, type] : llvm::enumerate(structType.getElementTypes())) {
      Value element = b.create<StructExtractOp>(loc, type, value, i);
      flattenAndUnpackIfStruct(b, loc, element, values);
    }
    return;
  }
  values.push_back(value);
}

Value KGEN::flattenAndPackIfStruct(mlir::RewriterBase &b, Location loc,
                                   Type type, ValueRange::iterator &it) {
  if (auto structType = dyn_cast<StructType>(type)) {
    SmallVector<Value> values;
    for (Type type : structType.getElementTypes())
      values.push_back(flattenAndPackIfStruct(b, loc, type, it));
    return b.create<StructCreateOp>(loc, structType, values);
  }
  return *it++;
}

void KGEN::flattenStructsInArguments(mlir::RewriterBase &b, Location loc,
                                     Block *body) {
  unsigned numArgs = body->getNumArguments();
  b.startOpModification(body->getParentOp());

  // Compute the flattened block argument types.
  SmallVector<Type> newTypes;
  SmallVector<Location> newLocs;
  for (BlockArgument arg : body->getArguments()) {
    unsigned curSize = newTypes.size();
    flattenTypeIfStruct(arg.getType(), newTypes);
    newLocs.append(newTypes.size() - curSize, arg.getLoc());
  }
  // Now add the new arguments.
  body->addArguments(newTypes, newLocs);

  // From the new arguments, reconstruct the old argument types.
  ValueRange origArgs = body->getArguments().slice(0, numArgs);
  SmallVector<Value> replacements;
  ValueRange::iterator it = origArgs.end();
  for (Type type : origArgs.getTypes())
    replacements.push_back(flattenAndPackIfStruct(b, loc, type, it));

  b.replaceAllUsesWith(origArgs, replacements);
  body->eraseArguments(0, numArgs);
  b.finalizeOpModification(body->getParentOp());
}

void KGEN::flattenStructsInOperands(mlir::RewriterBase &b, Operation *op) {
  // Flatten structs into the operand list.
  SmallVector<Value> newOperands;
  for (Value value : op->getOperands())
    flattenAndUnpackIfStruct(b, op->getLoc(), value, newOperands);
  b.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
}

Operation *KGEN::flattenStructsInResults(mlir::RewriterBase &b, Operation *op) {
  SmallVector<Type> types;
  for (Type type : op->getResultTypes())
    flattenTypeIfStruct(type, types);

  OperationState state(op->getLoc(), op->getName(), op->getOperands(), types);
  state.attributes = op->getAttrDictionary();
  for (Region &region : op->getRegions())
    state.addRegion()->takeBody(region);

  Operation *newOp = b.create(state);
  ValueRange::iterator it = ValueRange(newOp->getResults()).begin();
  SmallVector<Value> replacements;
  for (Type type : op->getResultTypes())
    replacements.push_back(flattenAndPackIfStruct(b, op->getLoc(), type, it));

  b.replaceOp(op, replacements);
  return newOp;
}
