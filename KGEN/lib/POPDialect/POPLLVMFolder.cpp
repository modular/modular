//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPOps.h"

#include "KGEN/Interpreter/InterpreterState.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace M;
using namespace KGEN;
using namespace POP;

// This attempts to lower the specified MLIR type into an LLVM IR rep.
static llvm::Type *convertToLLVM(Type type, llvm::LLVMContext &llvmCtx,
                                 TargetInfoAttr target) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return llvm::Type::getIntNTy(llvmCtx, intType.getWidth());

  if (isa<IndexType>(type)) {
    if (!target)
      return {};
    auto indexSize = DataLayoutInterface::getTypeAllocSize(target, type);
    assert(indexSize && "couldn't get the size of index?");
    return llvm::Type::getIntNTy(llvmCtx, *indexSize * 8);
  }

  return {};
}

// This attempts to lower the specified operand value into an LLVM IR
// representation that can be passed to a llvm::CallInst.
static llvm::Value *convertToLLVM(TypedAttr attr, llvm::LLVMContext &llvmCtx,
                                  TargetInfoAttr target) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    llvm::Type *type = convertToLLVM(attr.getType(), llvmCtx, target);
    if (!type)
      return {};

    return llvm::ConstantInt::get(type, intAttr.getValue());
  }

  // Otherwise we don't know what it is.
  return {};
}

/// Get the declaration of an overloaded llvm intrinsic. First we get the
/// overloaded argument types and/or result type from the CallIntrinsicOp, and
/// then use those to get the correct declaration of the overloaded intrinsic.
static llvm::Function *
getOverloadedDeclaration(ArrayRef<llvm::Type *> operandTypes,
                         llvm::Type *resType, llvm::Intrinsic::ID id,
                         llvm::Module *module) {
  // ATM we do not support variadic intrinsics.
  llvm::FunctionType *ft =
      llvm::FunctionType::get(resType, operandTypes, false);

  SmallVector<llvm::Intrinsic::IITDescriptor, 8> table;
  getIntrinsicInfoTableEntries(id, table);
  ArrayRef<llvm::Intrinsic::IITDescriptor> tableRef = table;

  SmallVector<llvm::Type *, 8> overloadedArgTys;
  if (llvm::Intrinsic::matchIntrinsicSignature(ft, tableRef,
                                               overloadedArgTys) !=
      llvm::Intrinsic::MatchIntrinsicTypesResult::MatchIntrinsicTypes_Match) {
    return {};
  }

  return llvm::Intrinsic::getDeclaration(module, id, overloadedArgTys);
}

template <typename T>
static std::string stringize(T value) {
  SmallVector<char> data;
  llvm::raw_svector_ostream os(data);
  os << value;
  return os.str().str();
}

// Interpreting an LLVM Intrinsic is a bit awkward.  We need to create an LLVM
// call operation, and then ask llvm to fold it for us.
ErrorTreeOrSuccess CallLLVMIntrinsicOp::interpret(ArrayRef<Attribute> operands,
                                                  InterpreterState &state) {
  // Check to see if we can resolve which intrinsic is being called.  If not,
  // then we can't fold it.
  auto name = dyn_cast<StringAttr>(getIntrinAttr());
  if (!name)
    return ErrorTree(getLoc(), "unknown intrinsic opcode");

  // See if LLVM knows what this is.
  llvm::Intrinsic::ID id = llvm::Function::lookupIntrinsicID(name.strref());
  if (!id)
    return ErrorTree(getLoc(),
                     "could not find LLVM intrinsic: '" + name.str() + "'");

  llvm::LLVMContext llvmContext;

  // Figure out the LLVM representation for all the operands.
  SmallVector<llvm::Value *> loweredOperands;
  for (auto v : operands) {
    // Try to understand what this value is.
    auto typedOp = ::dyn_cast<TypedAttr>(v);
    if (!typedOp)
      return ErrorTree(getLoc(), "LLVM intrinsic call has unknown operand: " +
                                     stringize(v));
    llvm::Value *loweredValue =
        convertToLLVM(typedOp, llvmContext, state.getTarget());
    if (!loweredValue)
      return ErrorTree(getLoc(), "LLVM intrinsic operand has unknown value: " +
                                     stringize(typedOp));
    loweredOperands.push_back(loweredValue);
  }

  // Compute the LLVM result type.
  if (getNumResults() == 0)
    return ErrorTree(getLoc(),
                     "cannot constant fold zero-result LLVM intrinsic: " +
                         name.str());
  if (getNumResults() != 1)
    return ErrorTree(getLoc(), "LLVM intrinsic operand has multiple results: " +
                                   name.str());
  llvm::Type *resultTy =
      convertToLLVM(getResult(0).getType(), llvmContext, state.getTarget());
  if (!resultTy)
    return ErrorTree(getLoc(), "LLVM intrinsic has unknown result type: " +
                                   stringize(getResult(0).getType()));

  // Try using "ConstantFoldBinaryIntrinsic" first - if it works, it avoids us
  // having to create a bunch of IR.
  llvm::Constant *result = nullptr;
  if (loweredOperands.size() == 2) {
    auto *lhs = ::dyn_cast<llvm::Constant>(loweredOperands[0]);
    auto *rhs = ::dyn_cast<llvm::Constant>(loweredOperands[1]);
    if (!lhs || !rhs)
      return ErrorTree(getLoc(), "LLVM intrinsic has non-constant operands");

    result = ConstantFoldBinaryIntrinsic(id, lhs, rhs, resultTy,
                                         /*FMFSource*/ nullptr);
  }

  if (!result) {
    // Otherwise, we handle this by creating a module with a call to the
    // intrinsic.
    llvm::Module module("folding", llvmContext);

    // Resolve the overloaded (or not) callee for the intrinsic call.
    llvm::Function *fn = nullptr;
    if (!llvm::Intrinsic::isOverloaded(id)) {
      fn = llvm::Intrinsic::getDeclaration(&module, id, {});
      assert(fn && "should always succeed");
    } else {
      SmallVector<llvm::Type *, 8> argTys;
      for (auto val : loweredOperands)
        argTys.push_back(val->getType());
      fn = getOverloadedDeclaration(argTys, resultTy, id, &module);
      if (!fn)
        return ErrorTree(
            getLoc(),
            "could not find overloaded declaration of LLVM intrinsic: " +
                name.str());
    }

    // Okay, we got the prototype for the intrinsic to call.  Generate a call to
    // it in another function.  We need a basic block to hold the call - just
    // abuse the intrinsic itself to own it.
    auto *block = llvm::BasicBlock::Create(llvmContext, Twine(), fn);

    auto *call =
        llvm::CallInst::Create(fn->getFunctionType(), fn, loweredOperands,
                               /*name*/ Twine(), block);

    // Now that we have a call, we can finally try to constant fold!
    // TODO: we aren't passing in a TargetLibraryInfo, which makes this super
    // conservative.
    result = llvm::ConstantFoldCall(call, fn, {}, /*TLI*/ nullptr);
  }

  if (!result)
    return ErrorTree(getLoc(),
                     "LLVM could not constant fold intrinsic: " + name.str());

  // If we got something back from LLVM, repackage it back up for MLIR to look
  // at.
  if (auto ci = ::dyn_cast<llvm::ConstantInt>(result)) {
    state.mapResults(IntegerAttr::get(getResult(0).getType(), ci->getValue()));
    return success();
  }

  return ErrorTree(getLoc(), "could not convert result of intrinsic: " +
                                 stringize(result));
}
