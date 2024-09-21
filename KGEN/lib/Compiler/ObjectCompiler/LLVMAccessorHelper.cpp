//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMAccessorHelper.h"

namespace {

// Helpers to access private field of llvm::MachineModuleInfo::MachineFunctions.
using MFAccessor = llvm::DenseMap<const llvm::Function *,
                                  std::unique_ptr<llvm::MachineFunction>>
    llvm::MachineModuleInfo::*;
MFAccessor getMFAccessor();
template <MFAccessor Instance>
struct RobberMFFromMachineModuleInfo {
  friend MFAccessor getMFAccessor() { return Instance; }
};
template struct RobberMFFromMachineModuleInfo<
    &llvm::MachineModuleInfo::MachineFunctions>;

// Helpers to access private field of llvm::MachineFunction::FunctionNumber.
using MFNumberAccessor = unsigned llvm::MachineFunction::*;
MFNumberAccessor getMFNumberAccessor();
template <MFNumberAccessor Instance>
struct RobberMFNumberFromMachineFunction {
  friend MFNumberAccessor getMFNumberAccessor() { return Instance; }
};
template struct RobberMFNumberFromMachineFunction<
    &llvm::MachineFunction::FunctionNumber>;

// Helpers to access private field of llvm::MachineModuleInfo::NextFnNum.
using NextFnNumAccessor = unsigned llvm::MachineModuleInfo::*;
NextFnNumAccessor getNextFnNumAccessor();
template <NextFnNumAccessor Instance>
struct RobberNextFnNumFromMachineModuleInfo {
  friend NextFnNumAccessor getNextFnNumAccessor() { return Instance; }
};
template struct RobberNextFnNumFromMachineModuleInfo<
    &llvm::MachineModuleInfo::NextFnNum>;

// Helpers to access private functions
template <typename Tag>
struct LLVMPrivateFnAccessor {
  /* export it ... */
  using type = typename Tag::type;
  static type ptr;
};

template <typename Tag>
typename LLVMPrivateFnAccessor<Tag>::type LLVMPrivateFnAccessor<Tag>::ptr;

template <typename Tag, typename Tag::type p>
struct LLVMPrivateFnAccessorRob : LLVMPrivateFnAccessor<Tag> {
  /* fill it ... */
  struct filler {
    filler() { LLVMPrivateFnAccessor<Tag>::ptr = p; }
  };
  static filler fillerObj;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
template <typename Tag, typename Tag::type p>
typename LLVMPrivateFnAccessorRob<Tag, p>::filler
    LLVMPrivateFnAccessorRob<Tag, p>::fillerObj;
#pragma GCC diagnostic pop

// Helpers to access private functions of llvm::MachineModuleInfo::NextFnNum.
struct MCContextGetSymbolEntryAccessor {
  using type = llvm::MCSymbolTableEntry &(llvm::MCContext::*)(llvm::StringRef);
};
template struct LLVMPrivateFnAccessorRob<MCContextGetSymbolEntryAccessor,
                                         &llvm::MCContext::getSymbolTableEntry>;

} // namespace

llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>> &
M::KGEN::getMachineFunctionsFromMachineModuleInfo(
    llvm::MachineModuleInfo &machineModuleInfo) {
  return std::invoke(getMFAccessor(), machineModuleInfo);
}

void M::KGEN::setMachineFunctionNumber(llvm::MachineFunction &mf,
                                       unsigned number) {
  unsigned &origNumber = std::invoke(getMFNumberAccessor(), mf);
  origNumber = number;
}

void M::KGEN::setNextFnNum(llvm::MachineModuleInfo &mmi, unsigned value) {
  unsigned &nextFnNum = std::invoke(getNextFnNumAccessor(), mmi);
  nextFnNum = value;
}

llvm::MCSymbolTableEntry &
M::KGEN::getMCContextSymbolTableEntry(llvm::StringRef name,
                                      llvm::MCContext &mcContext) {
  return (mcContext.*
          LLVMPrivateFnAccessor<MCContextGetSymbolEntryAccessor>::ptr)(name);
}
