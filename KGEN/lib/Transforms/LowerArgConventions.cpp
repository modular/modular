//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass performs the lowering of argument input conventions of concrete
// functions. This pass must run before inlining, but after elaboration. This
// pass will:
//
// 1. Move register passable types passed as `{owned,borrowed}_in_mem` to be
//    passed in register.
// 2. Promote register passable `byref_result` arguments to function results.
//    - This also handles functions that throw.
// 3. Unpacks kgen.pack typed arguments.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERARGCONVENTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerArgConventionsPass
    : KGEN::impl::LowerArgConventionsBase<LowerArgConventionsPass> {
  void runOnOperation() override;
};
} // namespace

/// Return the lowered type for an in-memory passed argument. If lowering is not
/// needed, return null.
static Type lowerPointerType(Type type) {
  // Only pointer types should be lowered.
  auto argPtr = dyn_cast<PointerType>(type);
  if (!argPtr)
    return {};

  // We don't lower memory-only structs.
  Type elType = argPtr.getElementType();
  if (auto structType = dyn_cast<StructType>(elType))
    if (structType.getIsMemoryOnly())
      return {};

  // We must be dealing with something register passable (e.g. index).
  return elType;
}

/// Return the pointer to the given type. For now, only support struct types
/// with a pointer, but it can be extended if needed.
static Type lowerTypeForGPU(Type type) {
  if (!isa<StructType>(type))
    return nullptr;

  bool hasPointer = false;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&hasPointer](PointerType ptr) {
    hasPointer = true;
    return nullptr;
  });
  replacer.replace(type);
  if (!hasPointer)
    return nullptr;

  return PointerType::get(type);
}

namespace {

enum ABI { Neither, ErrorOnly, ValueOnly, Both };
struct TransformResult {
  SmallVector<Type> newResultTypes;
  SmallVector<ArgConvention> newArgConventions;
  int abiLowering = ABI::Neither;
};
class Transform {
public:
  Transform(TargetInfoAttr target, DebugInfo::DISubprogramAttr spAttr)
      : target(target), spAttr(spAttr) {}
  virtual ~Transform() = default;
  virtual Type typeOfValueAt(unsigned operandIndex) = 0;
  virtual void performResultTransform(TransformResult const &result,
                                      unsigned operandIndex,
                                      Type loweredType) = 0;
  /// respond to a transform from PTR<X> to X
  virtual void applyPointerTransform(unsigned operandIndex, Type elType) = 0;
  /// respond to a transform from PACK<X,Y> to X,Y
  virtual void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                                  PackType type) = 0;
  /// respond to a transform from X to PTR<X>
  virtual void applyValueTransform(unsigned operandIndex, Type ptrType) = 0;
  Location addDI(Location loc);
  TargetInfoAttr target;
  DebugInfo::DISubprogramAttr spAttr;
};
class CallsiteTransform : public Transform {
public:
  CallsiteTransform(ImplicitLocOpBuilder &b, Operation *callOp,
                    TargetInfoAttr target, DebugInfo::DISubprogramAttr spAttr)
      : Transform(target, spAttr), b(b), callOp(callOp) {}
  Type typeOfValueAt(unsigned operandIndex) override;
  void performResultTransform(TransformResult const &result,
                              unsigned operandIndex, Type loweredType) override;
  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          PackType type) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  ImplicitLocOpBuilder &b;
  Operation *callOp;
  Value errorOperand;
  Value resultOperand;
};

class SignatureTransform : public Transform {
public:
  SignatureTransform(FuncType oldSig, TargetInfoAttr target,
                     DebugInfo::DISubprogramAttr spAttr)
      : Transform(target, spAttr), oldSig(oldSig) {
    llvm::append_range(newInputs, oldSig.getValues().getInputs());
  }
  Type typeOfValueAt(unsigned operandIndex) override;
  void performResultTransform(TransformResult const &result,
                              unsigned operandIndex, Type loweredType) override;
  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          PackType type) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  FuncType oldSig;
  SmallVector<Type> newInputs;
};

class FuncTransform : public Transform {
public:
  FuncTransform(ImplicitLocOpBuilder &b, FuncOp funcOp, TargetInfoAttr target);
  void performResultTransform(TransformResult const &result,
                              unsigned operandIndex, Type loweredType) override;
  Type typeOfValueAt(unsigned operandIndex) override;
  void applyPointerTransform(unsigned operandIndex, Type elType) override;
  void applyPackTransform(unsigned operandIndex, ArrayRef<Type> types,
                          PackType type) override;
  void applyValueTransform(unsigned operandIndex, Type ptrType) override;
  ImplicitLocOpBuilder &b;

  Block &block;
  Value newResPtr;
  Value newErrPtr;

private:
  void applyOneToOneTransform(unsigned operandIndex, Type newType,
                              llvm::function_ref<Value(Location, Value)> apply);
};
} // namespace

static void
insertAndUpdateConventions(SmallVectorImpl<ArgConvention> &conventions,
                           unsigned argConventionIndex, ArrayRef<Type> types,
                           int depth) {
  if (types.size() == 0) {
    conventions[argConventionIndex] = ArgConvention::ReadReg;
    return;
  }
  unsigned packSize = types.size();
  conventions.resize(conventions.size() + packSize - 1);
  for (unsigned i = conventions.size() - 1; i >= argConventionIndex + packSize;
       i--)
    conventions[i] = conventions[i - (packSize - 1)];
  for (unsigned i = 0; i < packSize; i++) {
    ArgConvention newConvention = ArgConvention::ReadReg;
    if (auto ptr = dyn_cast<PointerType>(types[i])) {
      // if the depth is 0 then this is a top level kgen.pack and we do not know
      // if it holds an address that is potentially written to.
      if (depth == 0)
        newConvention = ArgConvention::Mut;
      else if (!isa<KGEN::NoneType>(ptr.getElementType()))
        newConvention = ArgConvention::ReadMem;
    }
    conventions[argConventionIndex + i] = newConvention;
  }
}

static void transformNonResultValue(Transform *transform, unsigned operandIndex,
                                    SmallVector<ArgConvention> &conventions,
                                    unsigned argConventionIndex,
                                    int depth = 0) {

  Type type = transform->typeOfValueAt(operandIndex);
  ArgConvention convention = conventions[argConventionIndex];
  bool needsGPUTransform =
      transform->target && isGPUTriple(transform->target.getTriple());

  /// LOWER PTR
  if (isa<PointerType>(type) && !(convention == ArgConvention::ReadMem ||
                                  convention == ArgConvention::OwnedMem))
    return;

  if (auto elType = lowerPointerType(type)) {
    // Do not promote if we're going to demote next iteration.
    if (needsGPUTransform && lowerTypeForGPU(elType))
      return;
    transform->applyPointerTransform(operandIndex, elType);
    conventions[argConventionIndex] =
        conventions[argConventionIndex] == ArgConvention::OwnedMem
            ? ArgConvention::OwnedReg
            : ArgConvention::ReadReg;
    transformNonResultValue(transform, operandIndex, conventions,
                            argConventionIndex, ++depth);
  }

  /// LOWER PACK
  auto packType = dyn_cast<PackType>(type);
  if (packType && !(convention == ArgConvention::OwnedReg ||
                    convention == ArgConvention::ReadReg))
    return;

  if (packType) {
    auto variadic = cast_or_null<VariadicAttr>(packType.getVariadic());
    assert(variadic && "expected variadic pack type");
    SmallVector<Type> types;
    for (auto member : variadic.getValues()) {
      Type memberType = member.getType();
      if (auto typeValue = dyn_cast<KGEN::TypeParamAttr>(member))
        memberType = typeValue.getMlirType();
      types.push_back(memberType);
    }
    transform->applyPackTransform(operandIndex, types, packType);
    insertAndUpdateConventions(conventions, argConventionIndex, types, depth);
    transformNonResultValue(transform, operandIndex, conventions,
                            argConventionIndex, ++depth);
  }

  /// LIFT REG (GPU ONLY)
  if (!needsGPUTransform)
    return;
  if (Type newArgTy = lowerTypeForGPU(type)) {
    transform->applyValueTransform(operandIndex, newArgTy);
    conventions[argConventionIndex] = ArgConvention::ReadMem;
  }
}

FuncTransform::FuncTransform(ImplicitLocOpBuilder &b, FuncOp funcOp,
                             TargetInfoAttr target)
    : Transform(target, funcOp.getSubprogramScope()), b(b),
      block(funcOp.getBodyRegion().front()) {}

Location Transform::addDI(Location loc) {
  if (!spAttr)
    return loc;
  return FusedLoc::get(loc.getContext(), loc, spAttr);
}

Type FuncTransform::typeOfValueAt(unsigned operandIndex) {
  return block.getArgument(operandIndex).getType();
}

void FuncTransform::performResultTransform(TransformResult const &result,
                                           unsigned operandIndex,
                                           Type loweredType) {
  Value argVal = block.getArgument(operandIndex);
  auto alloc = b.create<POP::StackAllocationOp>(addDI(argVal.getLoc()),
                                                argVal.getType());
  argVal.replaceAllUsesWith(alloc);
  block.eraseArgument(operandIndex);
  if (result.abiLowering == ErrorOnly)
    newErrPtr = alloc;
  else if (result.abiLowering == Both || result.abiLowering == ValueOnly)
    newResPtr = alloc;
}

void FuncTransform::applyOneToOneTransform(
    unsigned operandIndex, Type newType,
    llvm::function_ref<Value(Location, Value)> apply) {
  auto point = b.saveInsertionPoint();
  auto resetState =
      llvm::make_scope_exit([&] { b.restoreInsertionPoint(point); });
  b.setInsertionPointToStart(&block);
  Location originalLocation = block.getArgument(operandIndex).getLoc();
  BlockArgument arg =
      block.insertArgument(operandIndex + 1, newType, originalLocation);
  Location location = addDI(block.getArgument(operandIndex).getLoc());
  auto image = apply(location, arg);
  block.getArgument(operandIndex).replaceAllUsesWith(image);
  block.eraseArgument(operandIndex);
}

void FuncTransform::applyPointerTransform(unsigned operandIndex, Type elType) {
  auto application = [&](Location location, Value newArg) -> Value {
    auto ptr = b.create<POP::StackAllocationOp>(
        location, PointerType::get(newArg.getType()));
    b.create<POP::StoreOp>(location, newArg, ptr);
    return ptr;
  };
  applyOneToOneTransform(operandIndex, elType, application);
}

void FuncTransform::applyValueTransform(unsigned operandIndex, Type ptrType) {
  auto application = [&](Location location, Value newArg) -> Value {
    return b.create<POP::LoadOp>(location, newArg).getResult();
  };
  applyOneToOneTransform(operandIndex, ptrType, application);
}

void FuncTransform::applyPackTransform(unsigned operandIndex,
                                       ArrayRef<Type> types, PackType type) {
  auto point = b.saveInsertionPoint();
  auto resetState =
      llvm::make_scope_exit([&] { b.restoreInsertionPoint(point); });
  Location originalLocation = block.getArgument(operandIndex).getLoc();
  b.setInsertionPointToStart(&block);
  SmallVector<Value> newArgs;
  unsigned curr = operandIndex;
  for (auto member : types)
    newArgs.push_back(block.insertArgument(++curr, member, originalLocation));
  auto pack =
      b.create<KGEN::PackCreateOp>(addDI(originalLocation), type, newArgs);
  block.getArgument(operandIndex).replaceAllUsesWith(pack);
  if (newArgs.empty())
    block.insertArgument(++curr, KGEN::NoneType::get(type.getContext()),
                         originalLocation);
  block.eraseArgument(operandIndex);
}

void CallsiteTransform::performResultTransform(TransformResult const &result,
                                               unsigned operandIndex,
                                               Type loweredType) {
  if (result.abiLowering == ErrorOnly)
    errorOperand = callOp->getOperand(operandIndex);
  else if (result.abiLowering == Both || result.abiLowering == ValueOnly)
    resultOperand = callOp->getOperand(operandIndex);
  callOp->eraseOperand(operandIndex);
}

void CallsiteTransform::applyPointerTransform(unsigned operandIndex,
                                              Type elType) {
  b.setInsertionPoint(callOp);
  Value arg = callOp->getOperands()[operandIndex];
  Value newArg = b.create<POP::LoadOp>(arg);
  callOp->setOperand(operandIndex, newArg);
}

void CallsiteTransform::applyValueTransform(unsigned operandIndex,
                                            Type ptrType) {
  b.setInsertionPoint(callOp);
  Value arg = callOp->getOperands()[operandIndex];
  Value newArg = b.create<POP::StackAllocationOp>(ptrType);
  b.create<POP::StoreOp>(arg, newArg);
  callOp->setOperand(operandIndex, newArg);
}

void CallsiteTransform::applyPackTransform(unsigned operandIndex,
                                           ArrayRef<Type> types,
                                           PackType type) {
  b.setInsertionPoint(callOp);
  Value operand = callOp->getOperands()[operandIndex];
  SmallVector<Value> newArgs;
  unsigned curr = 0;
  for (auto member : types) {
    newArgs.push_back(b.create<KGEN::PackExtractOp>(
        member, operand, IntegerAttr::get(b.getIndexType(), curr++)));
  }
  SmallVector<Value> newOperands;
  for (unsigned i = 0; i < operandIndex; i++)
    newOperands.push_back(callOp->getOperand(i));

  if (types.empty())
    newOperands.push_back(b.create<ParamConstantOp>(b.getAttr<NoneAttr>()));
  else
    llvm::append_range(newOperands, newArgs);

  for (unsigned i = operandIndex + 1; i < callOp->getNumOperands(); i++)
    newOperands.push_back(callOp->getOperand(i));
  callOp->setOperands(newOperands);
}

Type CallsiteTransform::typeOfValueAt(unsigned operandIndex) {
  return callOp->getOperandTypes()[operandIndex];
}

void SignatureTransform::performResultTransform(TransformResult const &result,
                                                unsigned operandIndex,
                                                Type loweredType) {
  newInputs.erase(newInputs.begin() + operandIndex);
}

void SignatureTransform::applyPointerTransform(unsigned operandIndex,
                                               Type elType) {
  newInputs[operandIndex] = elType;
}

void SignatureTransform::applyValueTransform(unsigned operandIndex,
                                             Type ptrType) {
  newInputs[operandIndex] = ptrType;
}

void SignatureTransform::applyPackTransform(unsigned operandIndex,
                                            ArrayRef<Type> types,
                                            PackType type) {
  if (types.empty()) {
    newInputs[operandIndex] = KGEN::NoneType::get(type.getContext());
    return;
  }
  auto erase_it = newInputs.begin() + operandIndex;
  newInputs.erase(erase_it);
  auto insert_it =
      newInputs.begin() + operandIndex; // Position where the erased element was
  newInputs.insert(insert_it, types.begin(), types.end());
}

Type SignatureTransform::typeOfValueAt(unsigned operandIndex) {
  return newInputs[operandIndex];
}

static TransformResult lowerSignature(FuncType oldSig,
                                      unsigned operandIndexInitial,
                                      Transform *transform) {
  unsigned operandIndex = operandIndexInitial;
  unsigned argConventionIndex = 0;

  TransformResult result;
  if (oldSig.isThrows())
    llvm::append_range(result.newResultTypes, oldSig.getResults());
  llvm::append_range(result.newArgConventions, oldSig.getArgConventions());
  SmallVector<ArgConvention> &argConventions = result.newArgConventions;
  while (argConventionIndex < argConventions.size()) {
    ArgConvention convention = argConventions[argConventionIndex];
    bool isResult = isResultSlot(convention);
    if (!isResult) {
      transformNonResultValue(transform, operandIndex, argConventions,
                              argConventionIndex);
    } else if (!oldSig.isAsync()) {
      if (Type loweredType =
              lowerPointerType(transform->typeOfValueAt(operandIndex))) {
        argConventions.erase(argConventions.begin() + argConventionIndex);
        result.abiLowering |= convention == ArgConvention::ByRefError
                                  ? ABI::ErrorOnly
                                  : ABI::ValueOnly;
        transform->performResultTransform(result, operandIndex, loweredType);
        switch (result.abiLowering) {
        case ABI::ErrorOnly:
          result.newResultTypes.push_back(loweredType);
          break;
        case ABI::Both: {
          Type errorType = result.newResultTypes.back();
          result.newResultTypes.clear();
          result.newResultTypes.push_back(
              VariantType::get({errorType, loweredType}));
        } break;
        case ABI::ValueOnly:
          result.newResultTypes.push_back(loweredType);
          break;
        default:
          break;
        }
        continue;
      }
    }
    argConventionIndex++;
    operandIndex++;
  }
  return result;
}

/// Lowers the given signature if needed
static FuncType lowerSignature(FuncType sig, TargetInfoAttr target,
                               DebugInfo::DISubprogramAttr spAttr) {
  SignatureTransform transform(sig, target, spAttr);
  TransformResult result = lowerSignature(sig, 0, &transform);
  FuncType newSig = FuncType::get(
      sig.getContext(),
      FunctionType::get(sig.getContext(), transform.newInputs,
                        result.newResultTypes.empty() ? sig.getResults()
                                                      : result.newResultTypes),
      result.newArgConventions, sig.getFnEffects(), sig.getMetadata());
  return newSig;
}

/// Helper to perform the bulk of the lowering for `kgen.call` and
/// `kgen.call_indirect` ops.
static void lowerCallOpImpl(Operation *op, FuncType oldSig,
                            DebugInfo::DISubprogramAttr spAttr) {

  ImplicitLocOpBuilder b(op->getLoc(), op);
  unsigned operandIndex = isa<CallIndirectOp>(op) ? 1 : 0;
  CallsiteTransform transform(b, op, lookupTargetInfo(op), spAttr);
  TransformResult result = lowerSignature(oldSig, operandIndex, &transform);
  int abiLowering = result.abiLowering;

  // Now update the result, if needed.
  if (abiLowering != Neither) {
    b.setInsertionPointAfter(op);
    OpResult res = op->getResult(0);
    if (oldSig.isThrows()) {
      // If the callee throws and both error and result were rewritten into a
      // variant, then we have to extract the relevant values from the variant.
      if (abiLowering == ABI::Both) {
        // Replace the i1 with a variant check.
        res.setType(result.newResultTypes[0]);
        auto isError = b.create<VariantIsOp>(res, 0);
        res.replaceAllUsesExcept(isError, isError);

        auto ifOp = b.create<HLCF::IfOp>(isError);
        b.createBlock(&ifOp.getThenRegion());
        b.create<POP::StoreOp>(b.create<VariantGetOp>(res, 0),
                               transform.errorOperand);
        b.create<HLCF::YieldOp>();

        b.createBlock(&ifOp.getElseRegion());
        b.create<POP::StoreOp>(b.create<VariantGetOp>(res, 1),
                               transform.resultOperand);
        b.create<HLCF::YieldOp>();
      } else {
        // In this case, we need to reallocate the operation with a different
        OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                             result.newResultTypes);
        state.attributes = op->getAttrDictionary();
        Operation *newOp = b.create(state);
        res.replaceAllUsesWith(newOp->getResult(0));

        // Store the relevant result in the branch in which it is known to have
        // a valid value.
        auto ifOp = b.create<HLCF::IfOp>(newOp->getResult(0));
        Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
        b.create<HLCF::YieldOp>();
        Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
        b.create<HLCF::YieldOp>();
        bool errorOnly = abiLowering == ErrorOnly;
        b.setInsertionPointToStart(errorOnly ? thenBlock : elseBlock);
        b.create<POP::StoreOp>(newOp->getResult(1),
                               errorOnly ? transform.errorOperand
                                         : transform.resultOperand);
        op->erase();
        op = newOp;
      }
    } else {
      // If the callee doesn't throw, we simply make every use take a new none.
      if (!res.use_empty()) {
        auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
        res.replaceAllUsesWith(none);
      }

      // Then just store the new callee result into the old memory result.
      res.setType(result.newResultTypes[0]);
      b.create<POP::StoreOp>(res, transform.resultOperand);
    }
  } else {
    result.newResultTypes.clear();
    llvm::append_range(result.newResultTypes, oldSig.getResults());
  }

  if (auto callOp = dyn_cast<CallOp>(op)) {
    FuncType newSig = FuncType::get(
        op->getContext(),
        FunctionType::get(op->getContext(), op->getOperandTypes(),
                          result.newResultTypes),
        result.newArgConventions, oldSig.getFnEffects(), oldSig.getMetadata());
    callOp.setCalleeAttr(SymbolConstantAttr::get(
        callOp.getCallee().getSymbol(), GeneratorType::get({}, newSig)));
  }
}

/// Emit IR for repacking the returned variant in the body of a throwing
/// function that we are currently lowering. This returns the new variant result
/// of the give type `newVariantTy`.
static Value repackFuncVariantResult(ReturnOp returnOp,
                                     VariantType newVariantTy, Value newResPtr,
                                     Value newErrPtr) {
  Value oldRetVal = returnOp.getOperand(0);
  ImplicitLocOpBuilder b(returnOp.getLoc(), returnOp);

  // We check the result is coming from. If we can guarantee that it's either an
  // error or not, we can just repack the error or the valid result.
  BoolAttr isError;
  if (mlir::matchPattern(oldRetVal, mlir::m_Constant(&isError))) {
    if (!isError.getValue()) {
      // This is guaranteed to be a normal return.
      return b.create<VariantCreateOp>(newVariantTy,
                                       b.create<POP::LoadOp>(newResPtr), 1);
    }
    // This is guaranteed to be an error return.
    return b.create<VariantCreateOp>(newVariantTy,
                                     b.create<POP::LoadOp>(newErrPtr), 0);
  }

  // We can't guarantee what the result is, so we emit conditional variant
  // repacking. We create an HCLF::IfOp, with a condition checking if there is
  // no error (i.e. the then branch will handle normal return). The result of
  // this IfOp is what we will return.
  auto ifOp = b.create<HLCF::IfOp>(newVariantTy, oldRetVal);

  // Populate the then branch (normal return).
  Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
  b.setInsertionPointToStart(thenBlock);
  Value thenRes = b.create<VariantCreateOp>(
      newVariantTy, b.create<POP::LoadOp>(newErrPtr), 0);
  b.create<HLCF::YieldOp>(thenRes);

  // Populate the else branch (error return).
  Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
  b.setInsertionPointToStart(elseBlock);
  Value elseRes = b.create<VariantCreateOp>(
      newVariantTy, b.create<POP::LoadOp>(newResPtr), 1);
  b.create<HLCF::YieldOp>(elseRes);

  return ifOp.getResult(0);
}

static void lowerFuncOp(FuncOp funcOp) {
  FuncType sig = funcOp.getFuncTypeGenerator().getBody();
  ImplicitLocOpBuilder b(funcOp.getLoc(), funcOp);
  b.setInsertionPoint(&funcOp.getBodyRegion().front().front());
  FuncTransform transform(b, funcOp, lookupTargetInfo(funcOp));
  TransformResult result = lowerSignature(sig, 0, &transform);
  FuncType newSig = FuncType::get(
      funcOp.getContext(),
      FunctionType::get(funcOp->getContext(),
                        funcOp.getBodyRegion().front().getArgumentTypes(),
                        result.newResultTypes.empty() ? sig.getResults()
                                                      : result.newResultTypes),
      result.newArgConventions, sig.getFnEffects(), sig.getMetadata());
  funcOp.setFuncTypeGenerator(
      GeneratorType::get(/*inputParamTypes=*/{}, newSig));
  if (result.abiLowering != Neither) {
    Block &body = funcOp.getBodyRegion().front();
    // Find all return sites in the function and rewrite them.
    body.walk([&](ReturnOp returnOp) {
      b.setInsertionPoint(returnOp);

      // If the function doesn't throw, we just load and return the new
      // result.
      if (!newSig.isThrows()) {
        auto newRes =
            b.create<POP::LoadOp>(returnOp.getLoc(), transform.newResPtr);
        returnOp.setOperand(0, newRes);
        return;
      }

      // If the function throws and we rewrote both the error and the
      // byref_result, we need to potentially unpack and repack the
      // result/error variant.
      if (result.abiLowering == Both) {
        auto newVariantTy = cast<VariantType>(newSig.getResults()[0]);
        Value newRetVal = repackFuncVariantResult(
            returnOp, newVariantTy, transform.newResPtr, transform.newErrPtr);
        returnOp.setOperand(0, newRetVal);
        return;
      }

      // Otherwise, we load either the error or the result, depending on which
      // got rewritten.
      Value toLoad =
          transform.newErrPtr ? transform.newErrPtr : transform.newResPtr;
      assert(toLoad && "should have been rewritten");
      Value newRes = b.create<POP::LoadOp>(returnOp.getLoc(), toLoad);
      returnOp->insertOperands(1, newRes);
    });
  }
}

void LowerArgConventionsPass::runOnOperation() {
  FuncOp func = getOperation();
  lowerFuncOp(func);

  // Lower the ops in the function body.
  DebugInfo::DISubprogramAttr spAttr = func.getSubprogramScope();
  func.walk([&](Operation *op) {
    if (auto callOp = dyn_cast<CallOp>(op))
      return lowerCallOpImpl(
          callOp, callOp.getCalleeSignature().getInstantiatedBody(), spAttr);
    if (auto callOp = dyn_cast<CallIndirectOp>(op))
      return lowerCallOpImpl(callOp, callOp.getCallee().getType().getBody(),
                             spAttr);
  });

  // We must do this in a second pass, otherwise ops like kgen.call_indirect
  // would be difficult to identify for lowering (since their argument types
  // would be lowered already).
  mlir::AttrTypeReplacer replacer;
  TargetInfoAttr target = lookupTargetInfo(func);
  replacer.addReplacement(
      [&](FuncType sig) { return lowerSignature(sig, target, spAttr); });
  auto metatype = TypeType::get(&getContext());
  replacer.addReplacement([&](TypeParamAttr type) {
    // Canonicalize metatypes.
    return TypeParamAttr::get(type.getMlirType(), metatype);
  });
  func.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  });
}
