//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Metal AIR LLVM Pass for Mojo GPU Compilation
//
// This pass automatically detects Metal GPU targets and transforms LLVM IR
// to be compatible with Apple's AIR (Apple Intermediate Representation).
//
//===----------------------------------------------------------------------===//

#include "MetalAIRPass.h"
#include "Target/max_license/Metal/MetalAddressSpace.h"
#include "LLVM/Transforms/PointerRewriter.h"
#include "LLVM/Transforms/TransformUtils.h"

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define KGEN_DEBUG_TYPE "metal-air-pass"

using namespace llvm;
using namespace M::KGEN;

// Attribute marking Metal kernel entry functions (defined in MetalLowering).
namespace M::KGEN {
extern const char *kMetalKernelAttr;
} // namespace M::KGEN

namespace {

// Forward declarations
void updateInstructionForAddressSpace(llvm::User *user, llvm::Value *oldPtr,
                                      llvm::Value *newPtr);
void updateGEPUser(llvm::User *user, llvm::GetElementPtrInst *oldGEP,
                   llvm::GetElementPtrInst *newGEP);

// Metal-specific constants

enum class MetalModuleFlagBehavior : unsigned {
  Error = Module::Error,
  Warning = Module::Warning,
  Require = Module::Require,
  Override = 7 // Custom behavior for AIR flags
};

enum class MetalCallingConvention : unsigned {
  C = CallingConv::C,
  PTX_Kernel = CallingConv::PTX_Kernel,
  AMDGPU_KERNEL = CallingConv::AMDGPU_KERNEL,
  SPIR_KERNEL = CallingConv::SPIR_KERNEL
};

struct ProjectVersion {
  unsigned major;
  unsigned minor;
  unsigned patch;
};

/// Versioning information for a Metal/AIR compilation target.
struct MetalVersionInfo {
  ProjectVersion metalVersion;
  ProjectVersion airVersion;
  std::string macOSVersion;
};

/// Parse an underscore-separated version string (e.g. "4_0" or "2_8_0").
/// The input StringRef must start at the first digit of the version.
static ProjectVersion parseVersion(StringRef version) {
  APInt val;
  unsigned major = 0, minor = 0, patch = 0;
  if (version.consumeInteger(10, val))
    llvm_unreachable("cannot parse 'major' version");
  major = val.getZExtValue();
  if (version.consume_front("_")) {
    if (version.consumeInteger(10, val))
      llvm_unreachable("cannot parse 'minor' version");
    minor = val.getZExtValue();
  }
  if (version.consume_front("_")) {
    if (version.consumeInteger(10, val))
      llvm_unreachable("cannot parse 'patch' version");
    patch = val.getZExtValue();
  }
  return {major, minor, patch};
}

/// Return the Metal/AIR/macOS version from the target-features string emitted
/// into the IR by LLVMLoweringUtils.
static MetalVersionInfo getMetalVersionFromFeatures(llvm::StringRef features) {
  ProjectVersion metalVersion = {3, 2, 0};
  ProjectVersion airVersion = {2, 7, 0};

  // Parse "+metal<MAJOR>_<MINOR>" feature flag (e.g. "+metal4_0").
  size_t metalIdx = features.find("+metal");
  if (metalIdx != StringRef::npos)
    metalVersion = parseVersion(features.substr(metalIdx + 6));

  // Parse "+air<MAJOR>_<MINOR>_<PATCH>" feature flag (e.g. "+air2_8_0").
  size_t airIdx = features.find("+air");
  if (airIdx != StringRef::npos)
    airVersion = parseVersion(features.substr(airIdx + 4));

  // Derive macOS SDK version from the AIR version.
  std::string macOSVersion =
      (airVersion.major > 2 || (airVersion.major == 2 && airVersion.minor >= 8))
          ? "26.0.0"
          : "15.0.0";

  return {metalVersion, airVersion, macOSVersion};
}

/// Return true if the target triple is `air64-`. Metal GPU targets use ARM64
/// during compilation, then get converted to AIR.
static bool isMetalTriple(const llvm::Triple &triple) {
  return StringRef(triple.str()).starts_with("air64-");
}

/// Check if this is a Metal GPU target
static bool isMetalTarget(Module &module) {
  // Check if module has metal-specific metadata or attributes indicating Metal
  // target. This is more reliable than just checking the triple since we use
  // arm64 as intermediate.

  // FIRST: Check for stored AIR intent metadata (most reliable)
  if (auto *airTargetFlag = module.getModuleFlag("metal.air_target_triple")) {
    if (auto *airTargetMD = dyn_cast<llvm::MDString>(airTargetFlag)) {
      llvm::Triple storedTriple(airTargetMD->getString());
      if (isMetalTriple(storedTriple))
        return true;
    }
  }

  // Look for existing air.* metadata
  if (module.getNamedMetadata("air.version") ||
      module.getNamedMetadata("air.kernel"))
    return true;

  // Check for air.kernel function attributes
  for (const Function &f : module)
    if (f.hasFnAttribute("air.kernel"))
      return true;

  // Check target triple for air64 (final Metal target)
  llvm::Triple triple(module.getTargetTriple());
  if (isMetalTriple(triple))
    return true;

  // For now, assume we're targeting Metal if we're on ARM64 macOS
  // This is a heuristic since we use arm64 as intermediate target
  if (triple.getArch() == llvm::Triple::aarch64 && triple.isMacOSX())
    return true;

  return false;
}

/// Store AIR intent and convert to compatible target for compilation
static bool transformTargetTriple(Module &module,
                                  const MetalVersionInfo &version) {
  const llvm::Triple &currentTriple = module.getTargetTriple();
  std::string currentTripleStr = currentTriple.str();

  // Check if this is intended for AIR/Metal output
  bool isAIRTarget = StringRef(currentTripleStr).starts_with("air64-") ||
                     isMetalTriple(currentTriple);

  if (isAIRTarget) {
    // Store AIR intent in module metadata for later restoration
    LLVMContext &ctx = module.getContext();
    std::string airTripleStr = "air64-apple-macosx" + version.macOSVersion;

    // Store the intended AIR target triple in metadata (only if not already
    // present)
    if (!module.getModuleFlag("metal.air_target_triple")) {
      module.addModuleFlag(Module::Error, "metal.air_target_triple",
                           MDString::get(ctx, airTripleStr));
    }

    // ALWAYS convert to compatible arm64 target for LLVM compilation
    // This prevents the target triple mismatch error with LLVM's TargetMachine
    std::string compatibleTriple = "arm64-apple-macosx" + version.macOSVersion;
    module.setTargetTriple(llvm::Triple(compatibleTriple));

    return true;
  }

  // Also handle cases where we're on ARM64 macOS and want Metal compilation
  if (currentTriple.getArch() == llvm::Triple::aarch64 &&
      currentTriple.isMacOSX()) {
    // This is already an ARM64 macOS target, but ensure we store Metal intent
    LLVMContext &ctx = module.getContext();
    std::string airTripleStr = "air64-apple-macosx" + version.macOSVersion;

    if (!module.getModuleFlag("metal.air_target_triple")) {
      module.addModuleFlag(Module::Error, "metal.air_target_triple",
                           MDString::get(ctx, airTripleStr));
    }
    return true;
  }
  return false;
}

/// Helper function to construct a new type identical to \p type with pointers
/// from `MetalAddressSpace::None` transformed to pointer from \p addressSpace
static llvm::Type *createTypeWithAddressSpace(llvm::Type *type,
                                              MetalAddressSpace addressSpace) {
  llvm::LLVMContext &ctx = type->getContext();
  std::function<llvm::Type *(llvm::Type *)> constructType =
      [&](llvm::Type *ty) -> llvm::Type * {
    if (auto *ptr = dyn_cast<llvm::PointerType>(ty)) {
      if (ptr->getAddressSpace() != MetalAddressSpace::None)
        return ptr;

      // Preserve pointer type if that's a typed pointer.
      if (auto *typedPtr = dyn_cast<llvm::TypedPointerType>(ty))
        return llvm::TypedPointerType::get(typedPtr->getElementType(),
                                           addressSpace);

      return llvm::PointerType::get(ctx, addressSpace);
    }

    if (auto *vectorTy = dyn_cast<llvm::VectorType>(ty)) {
      return llvm::VectorType::get(constructType(vectorTy->getElementType()),
                                   vectorTy->getElementCount());
    }

    if (!ty->isAggregateType())
      return ty;

    SmallVector<llvm::Type *, 4> subTypes;
    // Recursively traverse all nested types and append address space to
    // all walked pointers.
    for (Type *subTy : ty->subtypes())
      subTypes.push_back(constructType(subTy));

    if (isa<llvm::StructType>(ty))
      return llvm::StructType::get(ctx, subTypes);

    if (auto *arrayTy = dyn_cast<llvm::ArrayType>(ty)) {
      return llvm::ArrayType::get(constructType(arrayTy->getArrayElementType()),
                                  arrayTy->getArrayNumElements());
    }
    llvm_unreachable("Unhandled aggregate type");
  };

  return constructType(type);
}

/// Helper function to construct a new type identical to \p value's type with
/// address space taken from a given \p ptr.
static llvm::Type *createTypeWithAddressSpace(llvm::Value *value,
                                              llvm::Type *ptrTy) {
  MetalAddressSpace addressSpace = MetalAddressSpace::Device;
  if (auto *ptrType = dyn_cast<llvm::PointerType>(ptrTy))
    addressSpace = (MetalAddressSpace)ptrType->getAddressSpace();

  return createTypeWithAddressSpace(value->getType(), addressSpace);
}

/// Configure module for Metal compatibility
static bool configureModuleForMetal(Module &module) {
  bool changed = false;

  // Set module to use older LLVM IR patterns that Metal understands
  // This helps with xcrun metal compatibility

  // get the data layout from the module
  const std::string dataLayout = module.getDataLayoutStr();
  // set this to metal.air_data_layout
  LLVMContext &ctx = module.getContext();
  if (!module.getModuleFlag("metal.air_data_layout")) {
    module.addModuleFlag(Module::Error, "metal.air_data_layout",
                         MDString::get(ctx, dataLayout));
  }

  // Use ARM64 data layout that's compatible with TargetMachine during
  // compilation The MetalAIRPass will handle the final Metal-specific
  // formatting
  const std::string arm64DataLayout = "e-m:o-i64:64-i128:128-n32:64-S128";

  if (module.getDataLayoutStr() != arm64DataLayout) {
    module.setDataLayout(arm64DataLayout);
    changed = true;
  }

  return changed;
}

/// Check and report pointer types for Metal compatibility
static bool checkMetalPointerCompatibility(Module &module) {
  // Metal AIR format requires typed pointers, not opaque pointers
  // LLVM 15+ defaults to opaque pointers, but Metal toolchain expects typed
  // pointers

  for (const Function &func : module) {
    for (const BasicBlock &block : func) {
      for (const Instruction &inst : block) {
        // Check for pointer types in instructions
        if (auto *ptrInst = dyn_cast<AllocaInst>(&inst)) {
          Type *allocatedType = ptrInst->getAllocatedType();
          if (allocatedType) {
            KGEN_DEBUG(0, {
              llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
              llvm::dbgs() << "alloca inst has type " << *allocatedType << '\n';
            });
          }
        }

        // Check instruction operands for pointer types
        for (unsigned i = 0, e = inst.getNumOperands(); i < e; ++i) {
          Value *operand = inst.getOperand(i);
          if (operand->getType()->isPointerTy()) {
            // in newer LLVM, all pointers are opaque by default
            // We can't easily distinguish without looking at usage context
            if (dyn_cast<llvm::PointerType>(operand->getType())) {
              // This is a pointer type - in newer LLVM this is always opaque
              KGEN_DEBUG(0, {
                llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
                llvm::dbgs() << "opaque pointer ' " << *operand << " found\n";
              });
            }
          }
        }
      }
    }
  }

  return false; // No changes made to module structure
}

/// Add AIR-specific metadata required by metallib
static bool addAIRMetadata(Module &module, const MetalVersionInfo &version) {
  LLVMContext &ctx = module.getContext();
  bool changed = false;

  // Add AIR version metadata (required by metallib) - add at the end to match
  // reference ordering We'll add this after all other metadata to match the
  // reference ordering

  // Add kernel metadata - this will be populated later when we process
  // functions
  if (!module.getNamedMetadata("air.kernel")) {
    module.getOrInsertNamedMetadata("air.kernel");
    changed = true;
  }

  // Add air.compile_options metadata
  if (!module.getNamedMetadata("air.compile_options")) {
    NamedMDNode *compileOptions =
        module.getOrInsertNamedMetadata("air.compile_options");
    compileOptions->addOperand(
        MDNode::get(ctx, ArrayRef<Metadata *>(MDString::get(
                             ctx, "air.compile.denorms_disable"))));
    compileOptions->addOperand(
        MDNode::get(ctx, ArrayRef<Metadata *>(MDString::get(
                             ctx, "air.compile.fast_math_disable"))));
    compileOptions->addOperand(
        MDNode::get(ctx, ArrayRef<Metadata *>(MDString::get(
                             ctx, "air.compile.framebuffer_fetch_enable"))));
    changed = true;
  }

  // Add air.language_version metadata.
  // The Metal language version is derived from the target accelerator:
  //   apple-m5 → Metal 4.0; all others → Metal 3.2.
  if (!module.getNamedMetadata("air.language_version")) {
    NamedMDNode *langVersion =
        module.getOrInsertNamedMetadata("air.language_version");
    SmallVector<Metadata *, 4> langVals;
    langVals.push_back(MDString::get(ctx, "Metal"));
    langVals.push_back((Metadata *)ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.metalVersion.major)));
    langVals.push_back((Metadata *)ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.metalVersion.minor)));
    langVals.push_back((Metadata *)ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.metalVersion.patch)));
    langVersion->addOperand(MDNode::get(ctx, langVals));
    changed = true;
  }

  // Update module source filename from .mojo to .metal
  std::string sourceFileName = module.getSourceFileName();
  std::string updatedSourceFileName = sourceFileName;
  if (sourceFileName.empty()) {
    updatedSourceFileName = "debug_gpu_status.metal";
  } else {
    // Change .mojo extension to .metal
    size_t lastDot = sourceFileName.find_last_of('.');
    if (lastDot != std::string::npos) {
      std::string extension = sourceFileName.substr(lastDot);
      if (extension == ".mojo")
        updatedSourceFileName = sourceFileName.substr(0, lastDot) + ".metal";
    }
  }

  // Update the module's source filename if it changed
  if (sourceFileName != updatedSourceFileName) {
    module.setSourceFileName(updatedSourceFileName);
    changed = true;
  }

  // Add air.source_file_name metadata
  if (!module.getNamedMetadata("air.source_file_name")) {
    NamedMDNode *sourceFile =
        module.getOrInsertNamedMetadata("air.source_file_name");
    sourceFile->addOperand(
        MDNode::get(ctx, MDString::get(ctx, updatedSourceFileName)));
    changed = true;
  }

  // Add llvm.ident metadata with Metal compiler version
  if (!module.getNamedMetadata("llvm.ident")) {
    NamedMDNode *ident = module.getOrInsertNamedMetadata("llvm.ident");
    ident->addOperand(MDNode::get(
        ctx, MDString::get(
                 ctx, "Apple metal version 32023.620 (metalfe-32023.620)")));
    changed = true;
  }

  // Add required module flags for Metal with correct format
  // Also remove problematic flags that cause Metal toolchain issues
  NamedMDNode *flags = module.getModuleFlagsMetadata();
  bool hasSDKVersion = false;
  SmallVector<MDNode *, 8> flagsToKeep;

  if (flags) {
    for (unsigned i = 0, e = flags->getNumOperands(); i < e; ++i) {
      MDNode *flag = flags->getOperand(i);
      if (flag && flag->getNumOperands() >= 2) {
        if (auto *flagName = dyn_cast<MDString>(flag->getOperand(1))) {
          StringRef name = flagName->getString();
          if (name == "SDK Version") {
            hasSDKVersion = true;
            flagsToKeep.push_back(flag);
          } else if (name == "Debug Info Version") {
            // Remove Debug Info Version flag - Metal kernels don't have this
            changed = true;
            continue; // Skip this flag
          } else {
            flagsToKeep.push_back(flag);
          }
        } else {
          flagsToKeep.push_back(flag);
        }
      } else {
        flagsToKeep.push_back(flag);
      }
    }

    // Rebuild module flags without the problematic ones
    if (changed) {
      flags->eraseFromParent();
      NamedMDNode *newFlags =
          module.getOrInsertNamedMetadata("llvm.module.flags");
      for (MDNode *flag : flagsToKeep)
        newFlags->addOperand(flag);
    }
  }

  if (!hasSDKVersion) {
    // Add SDK Version flag in Metal-compatible format (constant array instead
    // of metadata array) Format: [2 x i32] [i32 15, i32 5] - matches working
    // Metal kernels
    Type *i32Ty = Type::getInt32Ty(ctx);
    ArrayType *arrayTy = ArrayType::get(i32Ty, 2);

    SmallVector<llvm::Constant *, 2> sdkVersionValues;
    sdkVersionValues.push_back(ConstantInt::get(i32Ty, 15));
    sdkVersionValues.push_back(ConstantInt::get(i32Ty, 5));
    llvm::Constant *sdkVersionArray =
        ConstantArray::get(arrayTy, sdkVersionValues);

    module.addModuleFlag(Module::Error, "SDK Version", sdkVersionArray);

    // Add wchar_size flag (Warning behavior like working Metal kernels)
    module.addModuleFlag(Module::Warning, "wchar_size", 4);

    // Add frame-pointer flag (Override behavior like working Metal kernels)
    module.addModuleFlag(Module::Override, "frame-pointer", 2);

    // Add AIR-specific flags with correct format (i32 7 for AIR flags)
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_device_buffers", 31);
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_constant_buffers", 31);
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_threadgroup_buffers", 31);
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_textures", 128);
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_read_write_textures", 8);
    module.addModuleFlag(
        static_cast<Module::ModFlagBehavior>(MetalModuleFlagBehavior::Override),
        "air.max_samplers", 16);

    changed = true;
  }

  // Add AIR version metadata (required by metallib) - add at the end to match
  // reference ordering. The AIR version is derived from the target accelerator:
  //   apple-m5 → AIR 2.8.0; all others → AIR 2.7.0.
  if (!module.getNamedMetadata("air.version")) {
    NamedMDNode *airVersion = module.getOrInsertNamedMetadata("air.version");
    SmallVector<Metadata *, 3> versionVals;
    versionVals.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.airVersion.major)));
    versionVals.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.airVersion.minor)));
    versionVals.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(ctx), version.airVersion.patch)));
    airVersion->addOperand(MDNode::get(ctx, versionVals));
    changed = true;
  }

  return changed;
}

/// Return true if function has `metal.kernel` attribute
static bool isKernelFunction(const Function &func) {
  // Skip declarations
  if (func.isDeclaration())
    return false;
  return func.hasFnAttribute(kMetalKernelAttr);
}

static const SmallVector<std::pair<std::string, llvm::Type *>>
getMetalArgsDescription(llvm::LLVMContext &ctx) {
  Type *int3VecTy = VectorType::get(Type::getInt32Ty(ctx), 3, false);
  SmallVector<std::pair<std::string, llvm::Type *>> metalArgsDescription{
      {"threadgroup_position_in_grid", int3VecTy},

      {"thread_position_in_threadgroup", int3VecTy},
      {"threads_per_threadgroup", int3VecTy},
      {"threads_per_grid", int3VecTy},
      // Corresponds to [[thread_index_in_simdgroup]] from Metal which has
      // uint type.
      {"thread_index_in_simdgroup", Type::getInt32Ty(ctx)},
  };
  return metalArgsDescription;
}

/// Return true if function is an AIR thread intrinsic
static bool isAIRThreadIntrinsic(llvm::Function &func) {
  StringRef name = func.getName();
  const SmallVector<std::pair<std::string, llvm::Type *>> &metalArgs =
      getMetalArgsDescription(func.getContext());
  for (auto [argName, _] : metalArgs) {
    if (!name.starts_with("air." + argName))
      continue;
    if (name == "air." + argName || name.ends_with(".x") ||
        name.ends_with(".y") || name.ends_with(".z"))
      return true;
  }
  return false;
} // namespace

namespace {

/// Analyze call graph to find functions that require thread parameters
class AIRThreadParameterUpdater : public CallGraphUpdater {
public:
  explicit AIRThreadParameterUpdater(llvm::Module &module,
                                     llvm::ModuleAnalysisManager &mam)
      : CallGraphUpdater(module, mam) {}

  /// Analyze entire call graph to find functions that require thread
  /// parameters:
  /// - Kernel functions
  /// - Functions that use air.thread* intrinsics
  /// - Functions that call functions that use air.thread* intrinsics
  virtual bool analyze() final {
    auto requiresThreadParams = [&](llvm::Function &f) {
      for (llvm::BasicBlock &bb : f) {
        for (llvm::Instruction &inst : bb) {
          if (llvm::CallInst *callInst =
                  llvm::dyn_cast<llvm::CallInst>(&inst)) {
            llvm::Function *callee = callInst->getCalledFunction();
            if (callee && (functionsToUpdate.contains(callee) ||
                           isAIRThreadIntrinsic(*callee))) {
              return true;
            }
          }
        }
      }
      return false;
    };

    SmallVector<llvm::Function *, 8> worklist;
    // Kernel functions and functions that use air.thread* intrinsics must be
    // transformed to include thread parameters
    for (llvm::Function &f : module) {
      if (isKernelFunction(f) || requiresThreadParams(f)) {
        functionsToUpdate.insert(&f);
        worklist.push_back(&f);
      }
    }

    // Analyze call graph and mark functions that require thread parameters iff
    // any function within its SCC requires thread parameters. Add all callers
    // to the worklist to add thread parameters.
    while (!worklist.empty()) {
      Function *f = worklist.pop_back_val();
      if (functionsToUpdate.contains(f))
        continue;

      for (llvm::LazyCallGraph::Node node :
           *graph.lookupSCC(*graph.lookup(*f))) {
        llvm::Function &fn = node.getFunction();
        functionsToUpdate.insert(&fn);

        // Since outer loop visits its SCC and adds function to the list of
        // functions to update, that's enough to only add one function of the
        // reference SCC.
        for (llvm::LazyCallGraph::SCC scc : *graph.lookupRefSCC(node)) {
          llvm::LazyCallGraph::Node sccNode = *scc.begin();
          worklist.push_back(&sccNode.getFunction());
        }
      }
    }
    return !functionsToUpdate.empty();
  }

  virtual llvm::Value *updateCall(llvm::CallInst &call, llvm::Function &newFunc,
                                  llvm::Function &callerFunc) final {
    SmallVector<llvm::Value *> args(call.args());
    const SmallVector<std::pair<std::string, llvm::Type *>> &metalArgs =
        getMetalArgsDescription(newFunc.getContext());
    for (size_t i = 0, e = metalArgs.size(); i < e; ++i)
      args.push_back(callerFunc.getArg(callerFunc.arg_size() - e + i));

    return CallInst::Create(&newFunc, args);
  }

  /// Create a new function with thread parameters and move the body of the \p
  /// func function into it.
  virtual llvm::Function *updateFunction(llvm::Function &func) final {
    llvm::LLVMContext &ctx = module.getContext();
    // Create new function signature with Metal address space annotations
    SmallVector<Type *, 8> newParamTypes;
    SmallVector<Attribute, 8> paramAttributes;

    for (const Argument &arg : func.args())
      newParamTypes.push_back(arg.getType());

    const SmallVector<std::pair<std::string, llvm::Type *>> &metalArgs =
        getMetalArgsDescription(ctx);

    for (unsigned i = 0, e = metalArgs.size(); i < e; ++i)
      newParamTypes.push_back(metalArgs[i].second);

    // Create new function type
    FunctionType *newFuncType =
        FunctionType::get(func.getReturnType(), newParamTypes, false);

    // Create new function
    Function *newFunc = Function::Create(newFuncType, func.getLinkage(),
                                         func.getName() + ".tmp", module);
    newFunc->copyAttributesFrom(&func);

    // Fix calling convention for Metal compatibility (must be C calling
    // convention) CRITICAL: Set this AFTER copyAttributesFrom to avoid it being
    // overwritten
    // FIXME: this code must be moved to a different location. It has nothing to
    // do with thread parameters.
    newFunc->setCallingConv(CallingConv::C);

    // Fix linkage for Metal compatibility
    newFunc->setLinkage(GlobalValue::ExternalLinkage);
    // Remove dso_local for AIR compatibility
    newFunc->setDSOLocal(false);
    newFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

    // Add attributes to track thread parameter indices
    newFunc->addFnAttr("metal.thread_args_added", "true");
    for (unsigned i = 0, e = metalArgs.size(); i < e; ++i) {
      std::string argName = metalArgs[i].first;
      unsigned idx = newParamTypes.size() - e + i;
      newFunc->addFnAttr("metal." + argName + "_idx", std::to_string(idx));
      newFunc->getArg(idx)->setName(argName);
    }
    // Add Metal-specific function attributes
    newFunc->addFnAttr("memory", "argmem: readwrite");

    // Move function body
    newFunc->splice(newFunc->begin(), &func);
    auto oldArgIt = func.arg_begin();
    auto newArgIt = newFunc->arg_begin();
    for (size_t i = 0, e = func.arg_size(); i < e;
         ++i, ++oldArgIt, ++newArgIt) {
      oldArgIt->replaceAllUsesWith(&*newArgIt);
    }
    return newFunc;
  }
};

// TODO: Remove this when new approach becomes stable.
class AddressSpacePropagationUpdater : public CallGraphUpdater {
private:
  DenseMap<llvm::Argument *, llvm::Type *> argTypeMap;

private:
  /// Update the arguments of \p newFunc using information from \p func
  void updateKernelFunctionArguments(llvm::Function &func,
                                     llvm::Function &newFunc) {
    llvm::LLVMContext &ctx = module.getContext();
    auto oldArgIt = func.arg_begin();
    auto newArgIt = newFunc.arg_begin();
    for (size_t i = 0, e = func.arg_size(); i < e;
         ++i, ++oldArgIt, ++newArgIt) {
      newArgIt->setName(oldArgIt->getName());

      // Add Metal-specific attributes for pointer parameters
      Type *argType = newArgIt->getType();
      if (argType->isPointerTy()) {
        // Add comprehensive Metal AIR buffer attributes
        newArgIt->addAttr(Attribute::get(ctx, "air-buffer-no-alias"));

        // Add standard LLVM attributes that Metal expects
        AttributeList attrs = newArgIt->getParent()->getAttributes();
        AttributeSet paramAttrs = attrs.getParamAttrs(newArgIt->getArgNo());

        // Build new attribute set with Metal-specific attributes
        AttrBuilder builder(ctx, paramAttrs);

        // Add memory safety attributes (use string versions for compatibility)
        builder.addAttribute("nocapture");
        builder.addAttribute("noundef");

        // Add read/write attributes based on parameter position
        if (i < 2) {
          // First two parameters are input buffers (readonly)
          builder.addAttribute("readonly");
        } else if (i == 2) {
          // Third parameter is output buffer (writeonly)
          builder.addAttribute("writeonly");
        }

        // Update the function with new attributes
        newFunc.addParamAttrs(newArgIt->getArgNo(), builder);
      }

      // Update all uses of old arguments with proper type casting for address
      // space changes, i64→i32 conversion, and struct extraction, and array
      // type conversion

      if (isa<llvm::StructType, llvm::ArrayType>(oldArgIt->getType())) {
        IRBuilder<> builder(&newFunc.getEntryBlock().front());
        auto *load = builder.CreateLoad(oldArgIt->getType(), &*newArgIt);
        oldArgIt->replaceAllUsesWith(load);
      } else if ((oldArgIt->getType()->isIntegerTy() ||
                  oldArgIt->getType()->isFloatingPointTy()) &&
                 newArgIt->getType()->isPointerTy()) {
        // Check if the old argument was a scalar that we converted to a
        // constant buffer pointer
        // Create a load instruction to get the scalar value from the constant
        // buffer
        BasicBlock &entryBB = newFunc.getEntryBlock();
        IRBuilder<> builder(entryBB.begin());

        // Load the scalar value from the constant buffer
        LoadInst *loadedValue = builder.CreateLoad(
            oldArgIt->getType(), &*newArgIt, oldArgIt->getName() + ".loaded");

        // Replace all uses of the old scalar argument with the loaded value
        // But first handle PHI nodes specially to avoid type mismatches
        SmallVector<PHINode *, 4> phiNodesToFix;
        SmallVector<User *, 8> otherUsers;

        for (User *user : oldArgIt->users()) {
          if (auto *phi = dyn_cast<PHINode>(user))
            phiNodesToFix.push_back(phi);
          else
            otherUsers.push_back(user);
        }

        // Replace non-PHI uses first
        for (User *user : otherUsers)
          for (Use &operand : user->operands())
            if (operand.get() == &*oldArgIt)
              operand.set(loadedValue);

        // Handle PHI nodes by updating their incoming values
        for (PHINode *phi : phiNodesToFix)
          for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e; ++i)
            if (phi->getIncomingValue(i) == &*oldArgIt)
              phi->setIncomingValue(i, loadedValue);
        continue;
      }
    }
  }
  /// Check if a type contains a pointer that still does not have address space.
  bool hasPointerWithoutAddressSpace(llvm::Type *ty) const {
    if (auto *ptr = dyn_cast<llvm::PointerType>(ty))
      return ptr->getAddressSpace() == MetalAddressSpace::None;

    for (llvm::Type *subTy : ty->subtypes())
      if (hasPointerWithoutAddressSpace(subTy))
        return true;
    return false;
  }

  /// Propagate address space on pointers within a function, including pointers
  /// that are part of the struct, arrays and vectors.
  /// FIXME: Revisit entire algorithm here as similar part is done in
  /// updateGEPUsers, however it may not visit all instructions.
  /// FIXME: The algorithm here should handle call to the same function but with
  /// different address spaces.
  /// FIXME: Consider to change it to 2 steps:
  ///   - Analysis of pointers to update by first assuming that all pointers
  ///     should be from `device` address space and update to other if needed.
  ///     That would require to correctly handle `insertvalue` and
  ///     `extractvalue`.
  ///   - Transformation of all pointers.
  void propagateAddressSpace(Function &func) {
    DenseSet<llvm::Instruction *> toRemove;
    SmallVector<llvm::Instruction *, 8> worklist;
    DenseSet<llvm::Instruction *> visited;

    ReversePostOrderTraversal<BasicBlock *> rpot(&func.getEntryBlock());
    for (llvm::BasicBlock *bb : rpot) {
      for (llvm::Instruction &inst : *bb) {
        llvm::Type *ty = inst.getType();
        if (!hasPointerWithoutAddressSpace(ty))
          continue;

        IRBuilder<> builder(&inst);
        // A list of all known instructions that can have their address space
        // propagated
        // FIXME: Constant values such as `undef`.
        Value *newValue =
            TypeSwitch<Instruction *, llvm::Value *>(&inst)
                .Case<llvm::LoadInst>([&](llvm::LoadInst *load) {
                  llvm::Type *newTy = createTypeWithAddressSpace(
                      load, load->getPointerOperand()->getType());
                  return builder.CreateLoad(newTy, load->getPointerOperand());
                })
                .Case<llvm::ExtractValueInst>(
                    [&](llvm::ExtractValueInst *extract) {
                      llvm::Value *agg = extract->getAggregateOperand();
                      ArrayRef<unsigned> idx = extract->getIndices();
                      return builder.CreateExtractValue(agg, idx);
                    })
                .Case<llvm::InsertValueInst>(
                    [&](llvm::InsertValueInst *insert) {
                      llvm::Value *agg = insert->getAggregateOperand();
                      llvm::Value *val = insert->getInsertedValueOperand();
                      if (isa<llvm::UndefValue>(agg)) {
                        agg = llvm::UndefValue::get(
                            createTypeWithAddressSpace(insert, val->getType()));
                      }
                      ArrayRef<unsigned> idx = insert->getIndices();
                      return builder.CreateInsertValue(agg, val, idx);
                    })
                .Case<llvm::GetElementPtrInst>(
                    [&](llvm::GetElementPtrInst *gep) {
                      llvm::GetElementPtrInst *newGEP =
                          cast<llvm::GetElementPtrInst>(builder.CreateGEP(
                              gep->getSourceElementType(),
                              gep->getPointerOperand(),
                              SmallVector<llvm::Value *>(gep->indices())));
                      newGEP->setIsInBounds(gep->isInBounds());
                      newGEP->setNoWrapFlags(gep->getNoWrapFlags());
                      return newGEP;
                    })
                .Case<llvm::BitCastInst>([&](llvm::BitCastInst *cast) {
                  llvm::Type *newTy =
                      createTypeWithAddressSpace(cast, cast->getSrcTy());
                  return builder.CreateBitCast(cast->getOperand(0), newTy);
                })
                .Case<llvm::PHINode>([&](llvm::PHINode *phi) {
                  // we expect that of incoming value has been updated, so it's
                  // possible to use its addrspace to create new PHI
                  llvm::Value *v = nullptr;
                  for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                       ++i) {
                    v = phi->getIncomingValue(i);
                    if (!hasPointerWithoutAddressSpace(v->getType()))
                      break;
                  }
                  assert(
                      v &&
                      "At least one incoming value must have address space.");
                  llvm::Type *newTy =
                      createTypeWithAddressSpace(phi, v->getType());
                  PHINode *newPhi =
                      builder.CreatePHI(newTy, phi->getNumIncomingValues());
                  // Need to create operands of the phi node in 2 steps: first
                  // step will add dummy values of the new type. Second step
                  // will set original value that might have different type
                  // until entire process is completed.
                  for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                       ++i) {
                    newPhi->addIncoming(UndefValue::get(newTy),
                                        phi->getIncomingBlock(i));
                  }
                  for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                       ++i) {
                    llvm::Value *val = phi->getIncomingValue(i);
                    if (val == phi)
                      val = newPhi;
                    newPhi->setOperand(i, val);
                  }
                  return newPhi;
                })
                .Case<llvm::AllocaInst>([&](llvm::AllocaInst *alloca) {
                  // FIXME: For now we allocate memory as if it's on device even
                  // though it should be a local memory. There should be a
                  // second pass that removes address space from alloca and all
                  // its users later.
                  return builder.CreateAlloca(alloca->getAllocatedType(),
                                              MetalAddressSpace::None,
                                              alloca->getArraySize());
                })
                .Case<llvm::SelectInst>([&](llvm::SelectInst *select) {
                  // Unlike other instructions, construction of the SelectInst
                  // doesn't take result type and takes it from the first
                  // operand, therefore to keep traversal simple if first
                  // operand was not updated yet, negate predicate, swap
                  // operands and construct SelectInst, otherwise simply
                  // construct SelectInst.
                  Value *cond = select->getCondition();
                  Value *trueVal = select->getTrueValue();
                  Value *falseVal = select->getFalseValue();
                  if (!hasPointerWithoutAddressSpace(trueVal->getType())) {
                    cond = builder.CreateNot(cond);
                    std::swap(trueVal, falseVal);
                  }
                  return builder.CreateSelect(cond, trueVal, falseVal);
                })
                .Case<llvm::ExtractElementInst>(
                    [&](llvm::ExtractElementInst *extract) {
                      return builder.CreateExtractElement(
                          extract->getOperand(0), extract->getOperand(1));
                    })
                .Default([&](Instruction *inst) {
                  llvm::dbgs() << *inst << '\n';
                  llvm_unreachable(
                      "Unhandled instruction that producer a pointer without "
                      "address space. Need to update propagateAddressSpace()");
                  return nullptr;
                });

        SmallVector<User *, 8> users(inst.users());
        for (User *user : users)
          for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
            if (user->getOperand(i) == &inst)
              user->setOperand(i, newValue);

        toRemove.insert(&inst);
      }
    }

    for (llvm::Instruction *inst : toRemove)
      inst->eraseFromParent();
  }

public:
  AddressSpacePropagationUpdater(llvm::Module &module,
                                 llvm::ModuleAnalysisManager &mam)
      : CallGraphUpdater(module, mam) {}

  // Find all functions that require address space propagation:
  // - Kernel functions
  // - Functions with pointer arguments without address space
  // in parallel construct new argument types map that will be used to update
  // functions.
  virtual bool analyze() final {
    LLVMContext &ctx = module.getContext();
    SmallVector<llvm::Function *> worklist;
    for (llvm::Function &func : module) {
      // Skip non-AIR declarations as we cannot update their body.
      // TODO: For now restrict to `air.atomic.global` intrinsic as it's known
      // to require addrspace.
      if (func.isDeclaration() &&
          !func.getName().starts_with("air.atomic.global."))
        continue;

      if (isKernelFunction(func)) {
        for (llvm::Argument &arg : func.args()) {
          llvm::Type *argType = arg.getType();
          llvm::Type *newArgType = nullptr;

          if (argType->isPointerTy()) {
            // Simple pointer - convert to device memory address space for Metal
            // kernels
            newArgType = llvm::PointerType::get(ctx, MetalAddressSpace::Device);

          } else if (argType->isIntegerTy() || argType->isFloatingPointTy()) {
            llvm::Attribute attr =
                func.getFnAttribute("metal.thread_index_in_simdgroup_idx");
            if (!attr.isValid() ||
                attr.getValueAsString() != std::to_string(arg.getArgNo())) {
              // Convert scalar types to constant buffer pointers for Metal
              // compatibility Metal expects scalars as constant buffer
              // parameters with buffer indices
              newArgType =
                  llvm::PointerType::get(ctx, MetalAddressSpace::Constant);
            }
          } else if (argType->isArrayTy() || argType->isStructTy()) {
            // Both these types have value semantics and Metal doesn't allow
            // them to be passed as values. Convert such arguments to pointers.
            newArgType = llvm::PointerType::get(ctx, MetalAddressSpace::Device);
          }
          if (newArgType)
            argTypeMap[&arg] = newArgType;
        }
        functionsToUpdate.insert(&func);
        worklist.push_back(&func);
      } else {
        // Non-kernel function requires transformation iff its argument is or
        // contains a pointer without address space
        bool requiresTransformation = false;
        for (Argument &arg : func.args()) {
          llvm::Type *argType = arg.getType();
          argType = argTypeMap.contains(&arg) ? argTypeMap[&arg] : argType;
          // Compare new type with old type to make a decision if transformation
          // is needed. Record new type to avoid recomputation
          llvm::Type *newType =
              createTypeWithAddressSpace(argType, MetalAddressSpace::Device);
          if (newType != arg.getType()) {
            argTypeMap[&arg] = newType;
            requiresTransformation = true;
          }
        }
        if (requiresTransformation)
          functionsToUpdate.insert(&func);
      }
    }

    DenseSet<llvm::Function *> visited;
    while (!worklist.empty()) {
      llvm::Function *func = worklist.pop_back_val();
      if (visited.contains(func))
        continue;
      llvm::LazyCallGraph::Node *node = graph.lookup(*func);

      // Add all callees to the worklist
      for (llvm::LazyCallGraph::Edge &edge : **node) {
        if (edge.getKind() == llvm::LazyCallGraph::Edge::Kind::Call) {
          llvm::Function *callee = &edge.getFunction();
          if (functionsToUpdate.contains(callee))
            continue;
          worklist.push_back(callee);
        }
      }
      visited.insert(func);
    }
    return !functionsToUpdate.empty();
  }

  virtual llvm::Function *updateFunction(llvm::Function &func) final {
    SmallVector<Type *, 8> newParamTypes;
    for (llvm::Argument &arg : func.args()) {
      auto it = argTypeMap.find(&arg);
      if (it != argTypeMap.end())
        newParamTypes.push_back(it->second);
      else
        newParamTypes.push_back(arg.getType());
    }

    // Create new function type
    FunctionType *newFuncType =
        FunctionType::get(func.getReturnType(), newParamTypes, false);

    // Create new function
    Function *newFunc = Function::Create(newFuncType, func.getLinkage(),
                                         func.getName() + ".tmp", module);
    newFunc->copyAttributesFrom(&func);

    if (func.isDeclaration())
      return newFunc;

    // Move function body
    newFunc->splice(newFunc->begin(), &func);

    if (isKernelFunction(*newFunc))
      updateKernelFunctionArguments(func, *newFunc);

    auto oldArgIt = func.arg_begin();
    auto newArgIt = newFunc->arg_begin();
    for (size_t i = 0, e = func.arg_size(); i < e;
         ++i, ++oldArgIt, ++newArgIt) {
      // Check if types match (they may differ due to address space annotations)
      if (oldArgIt->getType() == newArgIt->getType()) {
        // Types match, direct replacement
        oldArgIt->replaceAllUsesWith(&*newArgIt);
      } else {
        SmallVector<llvm::User *, 8> users(oldArgIt->users());
        // Types differ due to address space or i64→i32 conversion
        if (oldArgIt->getType()->isPointerTy() &&
            newArgIt->getType()->isPointerTy()) {
          for (User *user : users)
            updateInstructionForAddressSpace(user, &*oldArgIt, &*newArgIt);
        } else {
          // Since not all functions were updated yet type may not match,
          // therefore forcefully replace operands in all users
          for (User *user : users)
            for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
              if (user->getOperand(i) == &*oldArgIt)
                user->setOperand(i, &*newArgIt);
        }
      }
    }
    // Propagate address space for the rest of the instructions.
    // FIXME: Consider to combine it with updateInstructionForAddressSpace.
    propagateAddressSpace(*newFunc);
    return newFunc;
  }

  /// Create new call instruction with a \p newFunc callee, inserting
  /// bitcast or addrspacecast instructions as needed to match the new
  /// parameter types.
  virtual llvm::Value *updateCall(llvm::CallInst &call, llvm::Function &newFunc,
                                  llvm::Function &callerFunc) final {
    llvm::FunctionType *newFuncTy = newFunc.getFunctionType();
    SmallVector<llvm::Value *> args;
    IRBuilder<> builder(&call);
    for (unsigned i = 0, e = call.arg_size(); i < e; ++i) {
      llvm::Value *arg = call.getArgOperand(i);
      llvm::Type *expectedTy = newFuncTy->getParamType(i);
      if (arg->getType() != expectedTy) {
        if (arg->getType()->isPointerTy() && expectedTy->isPointerTy()) {
          assert(arg->getType()->getPointerAddressSpace() !=
                     expectedTy->getPointerAddressSpace() &&
                 "Pointer types differ but address spaces match");
          arg = builder.CreateAddrSpaceCast(arg, expectedTy);
        } else {
          assert(CastInst::castIsValid(Instruction::BitCast, arg, expectedTy) &&
                 "Argument type cannot be safely bitcast to expected type");
          arg = builder.CreateBitOrPointerCast(arg, expectedTy);
        }
      }
      args.push_back(arg);
    }
    return CallInst::Create(&newFunc, args);
  }
};

/// Inter-procedural address space propagation replacing
/// AddressSpacePropagationUpdater.
/// Analysis step: builds a Value→Type map
///          by seeding kernel args and propagating through def-use chains and
///          call sites.
/// Rewrite step: rewrites IR from the pre-built map using CallGraphUpdater.
class InterproceduralAddressSpacePropagationUpdater : public CallGraphUpdater {
private:
  /// Maps each Value to its new LLVM type after addrspace propagation.
  DenseMap<llvm::Value *, llvm::Type *> valueTypeMap;

  /// Maps alloca instructions to their new element type; the alloca result
  /// stays ptr addrspace(0) unless there's a store of a pointer with non-zero
  /// addrspace into that alloca
  DenseMap<AllocaInst *, llvm::Type *> allocaElemTypeMap;

  /// Maps functions to their new return type after address space promotion.
  DenseMap<llvm::Function *, llvm::Type *> returnTypeMap;

  /// Tracks values already processed in the current analyze() worklist.
  DenseSet<llvm::Value *> visited;

  bool hasPointerWithoutAddressSpace(llvm::Type *ty) const {
    if (auto *ptr = dyn_cast<llvm::PointerType>(ty))
      return ptr->getAddressSpace() == MetalAddressSpace::None;
    for (llvm::Type *subTy : ty->subtypes()) {
      if (hasPointerWithoutAddressSpace(subTy))
        return true;
    }
    return false;
  }

  /// Return a copy of aggregate type `aggTy` with the element reached by
  /// `indices` replaced by `newFieldTy`. Handles nested struct/array paths.
  llvm::Type *buildAggregateTypeWithField(llvm::Type *aggTy,
                                          ArrayRef<unsigned> indices,
                                          llvm::Type *newFieldTy) {
    if (indices.empty())
      return newFieldTy;

    llvm::LLVMContext &ctx = aggTy->getContext();
    unsigned idx = indices[0];
    ArrayRef<unsigned> rest = indices.drop_front();

    if (auto *st = dyn_cast<llvm::StructType>(aggTy)) {
      SmallVector<Type *, 8> fields(st->elements());
      if (idx >= fields.size())
        return nullptr;
      llvm::Type *newField =
          buildAggregateTypeWithField(fields[idx], rest, newFieldTy);
      if (!newField)
        return nullptr;
      fields[idx] = newField;
      return llvm::StructType::get(ctx, fields);
    }

    if (auto *at = dyn_cast<ArrayType>(aggTy)) {
      llvm::Type *newElem =
          buildAggregateTypeWithField(at->getElementType(), rest, newFieldTy);
      if (!newElem)
        return nullptr;
      return ArrayType::get(newElem, at->getNumElements());
    }

    return nullptr;
  }

  /// Merge two types by taking the address space from type where it's non-zero
  /// Return nullptr if address spaces are non-zeros and are different
  llvm::Type *mergePromotedTypes(llvm::Type *a, llvm::Type *b) {
    if (a == b)
      return a;
    if (!a || !b)
      return nullptr;
    llvm::LLVMContext &ctx = a->getContext();

    if (a->isPointerTy() && b->isPointerTy()) {
      unsigned asA = a->getPointerAddressSpace();
      unsigned asB = b->getPointerAddressSpace();
      if (asA == asB)
        return a;
      // Promote whichever is still addrspace(0)/None.
      if (asA == MetalAddressSpace::None)
        return b;
      if (asB == MetalAddressSpace::None)
        return a;
      // Two distinct non-zero address spaces: genuine conflict.
      return nullptr;
    }

    if (auto *stA = dyn_cast<llvm::StructType>(a)) {
      auto *stB = dyn_cast<llvm::StructType>(b);
      if (!stB || stA->getNumElements() != stB->getNumElements())
        return nullptr;
      SmallVector<llvm::Type *, 8> fields;
      bool changed = false;
      for (unsigned i = 0; i < stA->getNumElements(); ++i) {
        llvm::Type *elem =
            mergePromotedTypes(stA->getElementType(i), stB->getElementType(i));
        if (!elem)
          return nullptr;
        if (elem != stA->getElementType(i))
          changed = true;
        fields.push_back(elem);
      }
      return changed ? llvm::StructType::get(ctx, fields) : a;
    }

    if (auto *arrA = dyn_cast<llvm::ArrayType>(a)) {
      auto *arrB = dyn_cast<llvm::ArrayType>(b);
      if (!arrB || arrA->getNumElements() != arrB->getNumElements())
        return nullptr;
      llvm::Type *elem =
          mergePromotedTypes(arrA->getElementType(), arrB->getElementType());
      if (!elem)
        return nullptr;
      return elem != arrA->getElementType()
                 ? llvm::ArrayType::get(elem, arrA->getNumElements())
                 : a;
    }

    // Non-pointer, non-aggregate: must be identical.
    return a == b ? a : nullptr;
  }

  /// Compute the new result type of `inst` given that its operand at
  /// `operandNo` now has type `newOperandTy`. Returns nullptr when no change
  /// is needed.
  llvm::Type *computeNewResultType(llvm::Instruction *inst, unsigned operandNo,
                                   llvm::Type *newOperandTy) {
    return TypeSwitch<Instruction *, llvm::Type *>(inst)
        .Case<LoadInst>([&](LoadInst *load) -> llvm::Type * {
          if (operandNo != LoadInst::getPointerOperandIndex())
            return nullptr;
          return createTypeWithAddressSpace(load, newOperandTy);
        })
        .Case<GetElementPtrInst>([&](GetElementPtrInst *gep) -> llvm::Type * {
          if (operandNo != gep->getPointerOperandIndex())
            return nullptr;
          unsigned as =
              cast<llvm::PointerType>(newOperandTy)->getAddressSpace();
          return llvm::PointerType::get(gep->getContext(), as);
        })
        .Case<BitCastInst>([&](BitCastInst *) -> llvm::Type * {
          if (operandNo != 0)
            return nullptr;
          return createTypeWithAddressSpace(cast<BitCastInst>(inst),
                                            newOperandTy);
        })
        .Case<ExtractValueInst>([&](ExtractValueInst *extract) -> llvm::Type * {
          if (operandNo != 0)
            return nullptr;
          llvm::Type *ty = newOperandTy;
          for (unsigned idx : extract->getIndices()) {
            if (auto *st = dyn_cast<llvm::StructType>(ty))
              ty = st->getElementType(idx);
            else if (auto *at = dyn_cast<ArrayType>(ty))
              ty = at->getElementType();
            else
              return nullptr;
          }
          return ty != extract->getType() ? ty : nullptr;
        })
        .Case<InsertValueInst>([&](InsertValueInst *insert) -> llvm::Type * {
          if (operandNo == 0) {
            // Result type == aggregate type per LLVM langref, but also
            // incorporate any already-known promotion of the inserted value so
            // that both propagation paths (aggregate operand and inserted
            // value) converge on the same fully-promoted type.
            llvm::Value *val = insert->getInsertedValueOperand();
            llvm::Type *insertedTy = valueTypeMap.lookup(val);
            if (!insertedTy)
              insertedTy = val->getType();
            llvm::Type *result = buildAggregateTypeWithField(
                newOperandTy, insert->getIndices(), insertedTy);
            return result ? result : newOperandTy;
          }
          if (operandNo == insert->getInsertedValueOperandIndex()) {
            // Determine the best-known aggregate base type for field
            // substitution. Using the stale aggregate-operand type yields a
            // partially-promoted struct that conflicts with the fully-promoted
            // type computed through the aggregate's own propagation path.
            //
            // Walk up the insertvalue chain until we find a value whose
            // promoted type is already recorded (an intermediate insertvalue
            // seeded by an earlier worklist step), or reach a struct/array
            // Argument whose fully-promoted body type we can compute directly.
            // This handles chains of arbitrary length where intermediate links
            // may not yet have map entries when a pointer-arg propagation
            // reaches the last link.
            //
            // If neither source is available, also check this instruction's
            // own mapped type (handles the case where the body type was seeded
            // directly into this instruction before the pointer-arg's
            // propagation ran).
            llvm::Type *aggTy = nullptr;
            for (llvm::Value *cur = insert->getAggregateOperand(); cur;) {
              if (llvm::Type *t = valueTypeMap.lookup(cur))
                if (t->isAggregateType()) {
                  aggTy = t;
                  break;
                }
              // Struct/array Argument: compute the promoted body type directly.
              if (auto *arg = dyn_cast<Argument>(cur);
                  arg && arg->getType()->isAggregateType()) {
                aggTy = createTypeWithAddressSpace(arg->getType(),
                                                   MetalAddressSpace::Device);
                break;
              }
              // Step up through insertvalue chains.
              if (auto *prev = dyn_cast<InsertValueInst>(cur))
                cur = prev->getAggregateOperand();
              else
                break;
            }
            if (!aggTy)
              if (llvm::Type *t = valueTypeMap.lookup(insert))
                if (t->isAggregateType())
                  aggTy = t;
            if (!aggTy)
              aggTy = insert->getAggregateOperand()->getType();
            return buildAggregateTypeWithField(aggTy, insert->getIndices(),
                                               newOperandTy);
          }
          return nullptr;
        })
        .Case<PHINode>([&](PHINode *) -> llvm::Type * { return newOperandTy; })
        .Case<SelectInst>([&](SelectInst *) -> llvm::Type * {
          if (operandNo == 1 || operandNo == 2)
            return newOperandTy;
          return nullptr;
        })
        .Case<ExtractElementInst>([&](ExtractElementInst *) -> llvm::Type * {
          if (operandNo != 0)
            return nullptr;
          auto *vecTy = dyn_cast<VectorType>(newOperandTy);
          return vecTy ? vecTy->getElementType() : nullptr;
        })
        .Case<InsertElementInst>(
            [&](InsertElementInst *insert) -> llvm::Type * {
              if (operandNo ==
                  0) // vector operand changed → result is new vector type
                return newOperandTy;
              if (operandNo ==
                  1) { // element value changed → build new vector type
                auto *vecTy =
                    dyn_cast<VectorType>(insert->getOperand(0)->getType());
                if (!vecTy)
                  return nullptr;
                return VectorType::get(newOperandTy, vecTy->getElementCount());
              }
              return nullptr;
            })
        .Default([](Instruction *) -> llvm::Type * { return nullptr; });
  }

  /// Record `newTy` for `v` in valueTypeMap.
  /// Return false if new type conflicts with previously constructed type
  bool recordNewType(llvm::Value *v, llvm::Type *newTy,
                     SmallVectorImpl<llvm::Value *> &worklist) {
    if (!newTy || newTy == v->getType())
      return true;
    auto [it, inserted] = valueTypeMap.try_emplace(v, newTy);
    if (!inserted) {
      if (it->second == newTy)
        return true;
      // Conflict: try to merge by taking the maximum address space per pointer
      // field. This happens when two independent propagation paths each promote
      // a different subset of pointer fields in the same aggregate type (e.g.
      // the same ptr arg is inserted at two different indices of the same
      // struct). The merged type is monotonically more promoted than either
      // input, so re-queuing terminates.
      llvm::Type *merged = mergePromotedTypes(it->second, newTy);
      if (!merged)
        return false; // Genuine structural conflict.
      if (merged == it->second)
        return true; // No improvement.
      it->second = merged;
      // Allow re-processing with the updated type even if already visited.
      visited.erase(v);
      worklist.push_back(v);
      return true;
    }
    KGEN_DEBUG(1, {
      llvm::dbgs() << KGEN_DEBUG_TYPE << "addrspace propagation: recording "
                   << *v->getType() << " -> " << *newTy << " for: " << *v
                   << '\n';
    });
    worklist.push_back(v);
    return true;
  }

  /// Run one forward/backward propagation pass over the worklist. Clears
  /// `visited` on entry. Returns false on type conflict.
  bool runWorklistPass(SmallVectorImpl<llvm::Value *> &worklist) {
    visited.clear();
    while (!worklist.empty()) {
      llvm::Value *v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;

      llvm::Type *newTy = valueTypeMap.lookup(v);
      assert(newTy && "Worklist value must have an entry in valueTypeMap");

      // Backward propagation: certain instructions require operands to share
      // the result type (bitcast can't cross addrspaces; insertvalue result ==
      // aggregate type; phi/select operands must match; insertelement vec
      // operand must match result type).
      if (auto *inst = dyn_cast<Instruction>(v)) {
        if (!TypeSwitch<Instruction *, bool>(inst)
                 .Case<BitCastInst>([&](BitCastInst *cast) {
                   Value *src = cast->getOperand(0);
                   if (!isa<llvm::Constant>(src) &&
                       !recordNewType(src, newTy, worklist))
                     return false;
                   return true;
                 })
                 .Case<InsertValueInst>([&](InsertValueInst *insert) {
                   Value *agg = insert->getAggregateOperand();
                   if (!isa<llvm::Constant>(agg) &&
                       !recordNewType(agg, newTy, worklist))
                     return false;
                   // Propagate updated element type to the inserted value
                   // unless it already has a mapping (existing mappings take
                   // precedence; rewriteFunctionBody will insert casts).
                   llvm::Type *elemTy = ExtractValueInst::getIndexedType(
                       newTy, insert->getIndices());
                   if (elemTy) {
                     Value *val = insert->getInsertedValueOperand();
                     if (!isa<llvm::Constant>(val) &&
                         !valueTypeMap.count(val) &&
                         !recordNewType(val, elemTy, worklist))
                       return false;
                   }
                   return true;
                 })
                 .Case<PHINode>([&](PHINode *phi) {
                   for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                        ++i) {
                     Value *inVal = phi->getIncomingValue(i);
                     if (!isa<llvm::Constant>(inVal) &&
                         !recordNewType(inVal, newTy, worklist))
                       return false;
                   }
                   return true;
                 })
                 .Case<SelectInst>([&](SelectInst *sel) {
                   if (!isa<llvm::Constant>(sel->getTrueValue()) &&
                       !recordNewType(sel->getTrueValue(), newTy, worklist))
                     return false;
                   if (!isa<llvm::Constant>(sel->getFalseValue()) &&
                       !recordNewType(sel->getFalseValue(), newTy, worklist))
                     return false;
                   return true;
                 })
                 .Case<InsertElementInst>([&](InsertElementInst *insert) {
                   Value *vec = insert->getOperand(0);
                   if (!isa<llvm::Constant>(vec) &&
                       !recordNewType(vec, newTy, worklist))
                     return false;
                   return true;
                 })
                 .Default([](Instruction *) { return true; }))
          return false;
      }

      for (llvm::Use &use : v->uses()) {
        llvm::User *user = use.getUser();

        // Track changed return types for updateFunction.
        if (auto *ret = dyn_cast<ReturnInst>(user)) {
          llvm::Function *func = ret->getParent()->getParent();
          auto [it, inserted] = returnTypeMap.try_emplace(func, newTy);
          if (!inserted && it->second != newTy) {
            // Merge two partial promotions of the return type.
            llvm::Type *merged = mergePromotedTypes(it->second, newTy);
            if (!merged)
              return false; // Genuine structural conflict.
            it->second = merged;
          }
          continue;
        }

        // Inter-procedural: propagate new arg types into callee parameters.
        if (auto *call = dyn_cast<CallInst>(user)) {
          llvm::Function *callee = call->getCalledFunction();
          if (!callee) {
            // Type-mismatched calls use a bitcast wrapper; strip it.
            callee = dyn_cast<Function>(
                call->getCalledOperand()->stripPointerCasts());
          }
          if (!callee)
            continue;
          if (callee->isDeclaration()) {
            if (!callee->getName().starts_with("air.atomic.global."))
              continue;
            functionsToUpdate.insert(callee);
            // Propagate pointer→pointer addrspace changes; skip scalar→pointer
            // promotions — intrinsic declarations keep their original ABI.
            unsigned argNo = use.getOperandNo();
            if (argNo < callee->arg_size()) {
              llvm::Argument *declArg = callee->getArg(argNo);
              if (newTy->isPointerTy() && declArg->getType()->isPointerTy() &&
                  !recordNewType(declArg, newTy, worklist))
                return false;
            }
            continue;
          }

          unsigned argNo = use.getOperandNo();
          if (argNo >= callee->arg_size())
            continue;

          // Only propagate pointer→pointer addrspace changes; scalar→pointer
          // (constant-buffer) promotions are kernel-entry-only.
          llvm::Argument *calleeArg = callee->getArg(argNo);
          if (!calleeArg->getType()->isPointerTy() && newTy->isPointerTy())
            continue;

          if (!recordNewType(calleeArg, newTy, worklist))
            return false;
          functionsToUpdate.insert(callee);
          continue;
        }

        auto *inst = dyn_cast<Instruction>(user);
        if (!inst)
          continue;

        // In forward propagation, alloca always stays in addrspace(0). This
        // may change if pointer with non-zero addrspace is stored into this
        // alloca. See code below.
        if (isa<AllocaInst>(inst))
          continue;

        // When a pointer with an updated address space is stored into an
        // alloca, propagate the new type to all loads from that alloca.
        // This handles patterns like:
        //   %slot = alloca ptr
        //   store ptr %kernel_arg, ptr %slot  ; kernel_arg ->ptr addrspace(1)
        //   %p = load ptr, ptr %slot          ; should become ptr addrspace(1)
        if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
          if (use.getOperandNo() == 0 && newTy->isPointerTy()) {
            auto *allocaPtr =
                dyn_cast<AllocaInst>(storeInst->getPointerOperand());
            if (allocaPtr) {
              auto [it, inserted] =
                  allocaElemTypeMap.try_emplace(allocaPtr, newTy);
              if (!inserted && it->second != newTy) {
                // Ambiguous address space: conflicting types stored into alloca
                return false;
              }
              for (auto &allocaUse : allocaPtr->uses())
                if (auto *load = dyn_cast<LoadInst>(allocaUse.getUser()))
                  if (!recordNewType(load, newTy, worklist))
                    return false;
            }
          }
          continue;
        }

        llvm::Type *newInstTy =
            computeNewResultType(inst, use.getOperandNo(), newTy);
        if (newInstTy && newInstTy != inst->getType() &&
            !recordNewType(inst, newInstTy, worklist))
          return false;
      }
    }
    return true;
  }

  /// Promote the callee-side reloads of a device-bearing blob that was passed
  /// to `callee` by pointer in argument `argNo`. The matching parameter aliases
  /// the caller's private alloca, so it correctly stays addrspace(0); only its
  /// same-size ptr-bearing aggregate reloads are promoted, to the address
  /// spaces the blob's fields actually carry (`blobType`).
  ///
  /// This is the interprocedural analog of the same-function
  /// `scanTypePunAllocas` scan. The general call-graph addrspace propagation
  /// cannot recover it: the value flowing into the callee is a private `alloca`
  /// pointer that is genuinely addrspace(0), and the device pointers live in
  /// the *contents* of the spilled blob -- reading them back is a memory
  /// type-pun, not the argument/value type promotion that propagation models.
  ///
  /// `blobSize` is the stored blob's byte size; a reload must cover the same
  /// bytes to be a genuine type-pun. Returns false on a type conflict.
  bool promoteCalleeBlobLoads(llvm::Function *callee, unsigned argNo,
                              uint64_t blobSize, llvm::Type *blobType,
                              SmallVectorImpl<llvm::Value *> &worklist,
                              bool &foundTypePun) {
    if (!callee || callee->isDeclaration() || argNo >= callee->arg_size())
      return true;
    // Internal-linkage only: promoting the callee's reloads rewrites them for
    // every caller -- sound within this module, but not for an externally
    // visible function. Deliberately NOT also `hasOneUse()`: the fused-concat
    // closure is shared across concat sites, and every caller passes the same
    // device-bearing blob, so a shared callee must still be promoted (a
    // single-use bail silently zeroes concat>=4 on Metal). A callee shared
    // between device- and non-device-blob callers would instead need
    // per-call-site specialization.
    if (!callee->hasLocalLinkage())
      return true;
    const llvm::DataLayout &dl = module.getDataLayout();
    llvm::Argument *param = callee->getArg(argNo);
    if (!param->getType()->isPointerTy())
      return true;

    // Walk the parameter's transitive pointer uses (through address-space-
    // preserving bitcasts) to the loads that reload the blob.
    SmallVector<llvm::Value *, 8> ptrWork{param};
    DenseSet<llvm::Value *> seen;
    bool promotedAny = false;
    while (!ptrWork.empty()) {
      llvm::Value *cur = ptrWork.pop_back_val();
      if (!seen.insert(cur).second)
        continue;
      for (llvm::User *user : cur->users()) {
        if (auto *bc = dyn_cast<BitCastInst>(user)) {
          ptrWork.push_back(bc);
          continue;
        }
        auto *load = dyn_cast<LoadInst>(user);
        if (!load || !load->getType()->isAggregateType() ||
            !hasPointerWithoutAddressSpace(load->getType()))
          continue;
        llvm::TypeSize loadSz = dl.getTypeAllocSize(load->getType());
        if (loadSz.isScalable() || loadSz.getFixedValue() != blobSize)
          continue;
        // Promote the reload to the source blob's actual per-field address
        // spaces (carried by `blobType` post-promotion), not blanket-Device, so
        // a threadgroup/constant pointer in the blob is reloaded in its own
        // space. Device is the fallback only for an opaque-byte type-pun whose
        // reload aggregate doesn't structurally match the blob.
        llvm::Type *newLoadTy = mergePromotedTypes(load->getType(), blobType);
        if (!newLoadTy)
          newLoadTy = createTypeWithAddressSpace(load->getType(),
                                                 MetalAddressSpace::Device);
        if (newLoadTy == load->getType())
          continue;
        if (!recordNewType(load, newLoadTy, worklist))
          return false;
        promotedAny = true;
      }
    }
    if (promotedAny) {
      foundTypePun = true;
      functionsToUpdate.insert(callee);
    }
    return true;
  }

  /// Scan all functions in functionsToUpdate for the kgen.variant<T, i1>
  /// alloca type-pun pattern and record any promoted load types into worklist.
  ///
  /// kgen.variant<T, i1> lowers to an opaque byte-array struct with no visible
  /// pointer fields. A callee unpacks T by storing the blob to an alloca and
  /// reloading as the transparent T struct; the ptr fields emerge as
  /// addrspace(0). This scan detects such alloca pairs (same-size opaque store
  /// + ptr-bearing load) and records the promoted load types. When the reload
  /// happens in a callee that received the alloca by pointer, the promotion is
  /// delegated interprocedurally to `promoteCalleeBlobLoads`.
  ///
  /// Returns false on type conflict, true otherwise. Sets foundTypePun to true
  /// if at least one load type was promoted.
  bool scanTypePunAllocas(SmallVectorImpl<llvm::Value *> &worklist,
                          bool &foundTypePun) {
    const llvm::DataLayout &dl = module.getDataLayout();
    for (llvm::Function *func : functionsToUpdate) {
      if (func->isDeclaration())
        continue;
      for (BasicBlock &bb : *func) {
        for (Instruction &inst : bb) {
          auto *alloca = dyn_cast<AllocaInst>(&inst);
          if (!alloca)
            continue;

          // Find the opaque-blob store size (if any) and collect loads of
          // ptr-bearing aggregates whose size matches.
          uint64_t opaqueBlobSize = 0;
          // The stored blob's promoted type (with its real per-field address
          // spaces); reloads are promoted to match it, not blanket-Device.
          llvm::Type *opaqueBlobType = nullptr;
          SmallVector<LoadInst *, 4> ptrStructLoads;

          // Calls that receive the alloca pointer by value; the callee
          // reloads the device-bearing blob and (without promotion) reads its
          // pointer fields as addrspace(0). Tracked so we can promote those
          // callee-side reloads interprocedurally (see below).
          SmallVector<std::pair<CallInst *, unsigned>, 4> escapingCalls;

          for (auto &use : alloca->uses()) {
            if (auto *store = dyn_cast<StoreInst>(use.getUser())) {
              // Only consider uses where the alloca is the store destination
              // (the pointer operand), not the value being stored.
              if (use.getOperandNo() != StoreInst::getPointerOperandIndex())
                continue;
              // Use the post-promotion type of the stored value if the main
              // worklist already promoted it (e.g. a
              // `_FusedInputVariadicTensors` whose addrspace(1) pointers were
              // inserted from kernel args). The original IR type is still
              // addrspace(0) at this point. Only the pointer-bearing case needs
              // this; a genuinely opaque blob has no valueTypeMap entry and
              // falls through with its original type.
              llvm::Value *storedVal = store->getValueOperand();
              llvm::Type *storeTy = storedVal->getType();
              if (hasPointerWithoutAddressSpace(storeTy))
                if (llvm::Type *promoted = valueTypeMap.lookup(storedVal))
                  storeTy = promoted;
              // Either a genuinely opaque blob (no visible pointers) or an
              // aggregate whose pointers are all already in a device-capable
              // address space — both are safe to reinterpret on the load side.
              if (storeTy->isAggregateType() &&
                  !hasPointerWithoutAddressSpace(storeTy)) {
                llvm::TypeSize sz = dl.getTypeAllocSize(storeTy);
                if (sz.isScalable()) {
                  opaqueBlobSize = 0;
                  break; // Scalable types can't be type-punned; skip alloca.
                }
                opaqueBlobSize = sz.getFixedValue();
                opaqueBlobType = storeTy;
              }
            } else if (auto *load = dyn_cast<LoadInst>(use.getUser())) {
              if (load->getType()->isAggregateType() &&
                  hasPointerWithoutAddressSpace(load->getType()))
                ptrStructLoads.push_back(load);
            } else if (auto *call = dyn_cast<CallInst>(use.getUser())) {
              // The alloca pointer is passed by value to a callee that reloads
              // the device-bearing blob on its own stack copy. This is the
              // `elementwise`/fused-variadic concat pattern: the input
              // `_FusedInputVariadicTensors` (holding addrspace(1) input ptrs)
              // is spilled to an alloca and handed to the outlined per-element
              // lambda by pointer. Without promotion the callee's reload yields
              // addrspace(0) input pointers -> reads zeros on Metal.
              if (llvm::Function *callee = call->getCalledFunction();
                  callee && !callee->isDeclaration())
                escapingCalls.emplace_back(call, use.getOperandNo());
            }
          }

          if (!opaqueBlobSize)
            continue;
          if (ptrStructLoads.empty() && escapingCalls.empty())
            continue;

          for (LoadInst *load : ptrStructLoads) {
            // Only treat as a type-pun when the load covers the same bytes as
            // the opaque store; a smaller load is a legitimate partial access.
            llvm::TypeSize loadSz = dl.getTypeAllocSize(load->getType());
            if (loadSz.isScalable() || loadSz.getFixedValue() != opaqueBlobSize)
              continue;
            // Promote to the blob's actual per-field address spaces (see
            // `promoteCalleeBlobLoads`); Device only as the opaque-blob
            // fallback.
            llvm::Type *newLoadTy =
                mergePromotedTypes(load->getType(), opaqueBlobType);
            if (!newLoadTy)
              newLoadTy = createTypeWithAddressSpace(load->getType(),
                                                     MetalAddressSpace::Device);
            if (newLoadTy != load->getType()) {
              if (!recordNewType(load, newLoadTy, worklist))
                return false;
              foundTypePun = true;
            }
          }

          // Interprocedural type-pun: promote the callee-side reloads of the
          // device-bearing blob passed by pointer. The callee parameter itself
          // stays addrspace(0) (it points at the caller's private alloca); only
          // its same-size ptr-bearing aggregate loads are promoted to the
          // device address space, mirroring the same-function scan above.
          for (auto [call, argNo] : escapingCalls) {
            if (!promoteCalleeBlobLoads(call->getCalledFunction(), argNo,
                                        opaqueBlobSize, opaqueBlobType,
                                        worklist, foundTypePun))
              return false;
          }
        }
      }
    }
    return true;
  }

  /// Add Metal buffer attributes to pointer parameters of a kernel function.
  void addKernelPointerAttributes(llvm::Function &func) {
    llvm::LLVMContext &ctx = module.getContext();

    // Thread parameters are the last N args; don't annotate them.
    unsigned metalArgStart = func.arg_size();
    for (auto &[name, _] : getMetalArgsDescription(ctx)) {
      Attribute attr = func.getFnAttribute("metal." + name + "_idx");
      if (attr.isValid()) {
        unsigned idx = std::stoul(attr.getValueAsString().str());
        metalArgStart = std::min(metalArgStart, idx);
      }
    }

    for (unsigned i = 0; i < metalArgStart; ++i) {
      llvm::Argument *arg = func.getArg(i);
      if (!arg->getType()->isPointerTy())
        continue;
      arg->addAttr(Attribute::get(ctx, "air-buffer-no-alias"));
      AttrBuilder builder(ctx);
      builder.addAttribute("nocapture");
      builder.addAttribute("noundef");
      // Match Metal's calling convention: first two buffer args are inputs
      // (readonly), third is the output (writeonly).
      if (i < 2)
        builder.addAttribute("readonly");
      else if (i == 2)
        builder.addAttribute("writeonly");
      func.addParamAttrs(i, builder);
    }
  }

  /// Rewrite instructions in `func` whose result types appear in valueTypeMap.
  /// Walks the function in RPOT so defs are processed before uses.
  void rewriteFunctionBody(llvm::Function &func) {
    // Maps old instruction → replacement for later uses still referencing it.
    DenseMap<Value *, Value *> replacedValues;
    SmallVector<Instruction *, 16> toErase;

    auto getVal = [&](Value *v) -> Value * {
      auto it = replacedValues.find(v);
      return it != replacedValues.end() ? it->second : v;
    };

    // Update alloca element types first to avoid iterator invalidation.
    SmallVector<AllocaInst *, 4> allocasToUpdate;
    for (BasicBlock &bb : func)
      for (Instruction &inst : bb)
        if (auto *alloca = dyn_cast<AllocaInst>(&inst))
          if (allocaElemTypeMap.count(alloca))
            allocasToUpdate.push_back(alloca);

    for (AllocaInst *alloca : allocasToUpdate) {
      llvm::Type *newElemTy = allocaElemTypeMap[alloca];
      IRBuilder<> allocaBuilder(alloca);
      AllocaInst *newAlloca =
          allocaBuilder.CreateAlloca(newElemTy, alloca->getAddressSpace(),
                                     alloca->getArraySize(), alloca->getName());
      newAlloca->setAlignment(alloca->getAlign());
      alloca->replaceAllUsesWith(newAlloca);
      // Erase from map before erasing the alloca to avoid stale-pointer reuse.
      allocaElemTypeMap.erase(alloca);
      alloca->eraseFromParent();
    }

    ReversePostOrderTraversal<BasicBlock *> rpot(&func.getEntryBlock());
    for (BasicBlock *bb : rpot) {
      for (Instruction &inst : *bb) {
        llvm::Type *newTy = valueTypeMap.lookup(&inst);
        if (!newTy)
          continue;

        IRBuilder<> builder(&inst);
        Value *newVal =
            TypeSwitch<Instruction *, Value *>(&inst)
                .Case<LoadInst>([&](LoadInst *load) -> Value * {
                  return builder.CreateLoad(newTy,
                                            getVal(load->getPointerOperand()));
                })
                .Case<GetElementPtrInst>(
                    [&](GetElementPtrInst *gep) -> Value * {
                      SmallVector<Value *, 4> indices;
                      for (Value *idx : gep->indices())
                        indices.push_back(getVal(idx));
                      auto *newGEP = cast<GetElementPtrInst>(builder.CreateGEP(
                          gep->getSourceElementType(),
                          getVal(gep->getPointerOperand()), indices));
                      newGEP->setIsInBounds(gep->isInBounds());
                      newGEP->setNoWrapFlags(gep->getNoWrapFlags());
                      return newGEP;
                    })
                .Case<BitCastInst>([&](BitCastInst *cast) -> Value * {
                  Value *src = getVal(cast->getOperand(0));
                  // If the source and result now have different address spaces,
                  // bitcast is invalid — use addrspacecast instead.
                  if (src->getType()->isPointerTy() && newTy->isPointerTy() &&
                      src->getType()->getPointerAddressSpace() !=
                          newTy->getPointerAddressSpace())
                    return builder.CreateAddrSpaceCast(src, newTy);
                  return builder.CreateBitCast(src, newTy);
                })
                .Case<ExtractValueInst>(
                    [&](ExtractValueInst *extract) -> Value * {
                      Value *result = builder.CreateExtractValue(
                          getVal(extract->getAggregateOperand()),
                          extract->getIndices());
                      // If the aggregate wasn't in valueTypeMap (e.g. loaded
                      // from a non-updated alloca), cast to the expected type.
                      if (result->getType() != newTy &&
                          result->getType()->isPointerTy() &&
                          newTy->isPointerTy())
                        result = builder.CreateAddrSpaceCast(result, newTy);
                      return result;
                    })
                .Case<InsertValueInst>([&](InsertValueInst *insert) -> Value * {
                  Value *agg = getVal(insert->getAggregateOperand());
                  Value *val = getVal(insert->getInsertedValueOperand());
                  if (isa<UndefValue>(agg))
                    agg = UndefValue::get(newTy);
                  else if (isa<ConstantAggregateZero>(agg))
                    agg = Constant::getNullValue(newTy);
                  else if (agg->getType() != newTy) {
                    // agg has a partially-promoted type (an earlier insertvalue
                    // in the chain promoted fewer fields). Rebuild it to newTy,
                    // copying each field with addrspace-casts as needed, but
                    // skipping the field being overwritten by this instruction.
                    auto *oldST = dyn_cast<llvm::StructType>(agg->getType());
                    auto *newST = dyn_cast<llvm::StructType>(newTy);
                    if (oldST && newST &&
                        oldST->getNumElements() == newST->getNumElements() &&
                        insert->getIndices().size() == 1) {
                      unsigned skipIdx = insert->getIndices()[0];
                      llvm::Value *rebuilt = UndefValue::get(newTy);
                      for (unsigned i = 0; i < newST->getNumElements(); ++i) {
                        if (i == skipIdx)
                          continue;
                        llvm::Value *field =
                            builder.CreateExtractValue(agg, {i});
                        llvm::Type *newFldTy = newST->getElementType(i);
                        if (field->getType() != newFldTy &&
                            field->getType()->isPointerTy() &&
                            newFldTy->isPointerTy())
                          field = builder.CreateAddrSpaceCast(field, newFldTy);
                        rebuilt =
                            builder.CreateInsertValue(rebuilt, field, {i});
                      }
                      agg = rebuilt;
                    }
                  }
                  // Use newTy for the field type so pointer addrspaces match.
                  llvm::Type *fieldTy = ExtractValueInst::getIndexedType(
                      newTy, insert->getIndices());
                  if (fieldTy && fieldTy != val->getType() &&
                      fieldTy->isPointerTy() && val->getType()->isPointerTy())
                    val = builder.CreateAddrSpaceCast(val, fieldTy);
                  return builder.CreateInsertValue(agg, val,
                                                   insert->getIndices());
                })
                .Case<PHINode>([&](PHINode *phi) -> Value * {
                  PHINode *newPhi =
                      builder.CreatePHI(newTy, phi->getNumIncomingValues());
                  // Two passes: first add dummy incoming values, then fix them.
                  for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                       ++i)
                    newPhi->addIncoming(UndefValue::get(newTy),
                                        phi->getIncomingBlock(i));
                  for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e;
                       ++i) {
                    Value *v = phi->getIncomingValue(i);
                    v = (v == phi) ? newPhi : getVal(v);
                    // Cast pointer constants left unchanged by getVal.
                    if (newTy->isPointerTy() && v->getType() != newTy &&
                        v->getType()->isPointerTy()) {
                      if (auto *c = dyn_cast<llvm::Constant>(v))
                        v = llvm::ConstantExpr::getAddrSpaceCast(c, newTy);
                    }
                    newPhi->setOperand(i, v);
                  }
                  return newPhi;
                })
                .Case<SelectInst>([&](SelectInst *sel) -> Value * {
                  Value *trueVal = getVal(sel->getTrueValue());
                  Value *falseVal = getVal(sel->getFalseValue());
                  // Cast pointer constants (e.g. null) that getVal left
                  // unchanged to avoid a CreateSelect type mismatch.
                  if (newTy->isPointerTy()) {
                    if (trueVal->getType() != newTy &&
                        trueVal->getType()->isPointerTy())
                      trueVal = builder.CreateAddrSpaceCast(trueVal, newTy);
                    if (falseVal->getType() != newTy &&
                        falseVal->getType()->isPointerTy())
                      falseVal = builder.CreateAddrSpaceCast(falseVal, newTy);
                  }
                  return builder.CreateSelect(sel->getCondition(), trueVal,
                                              falseVal);
                })
                .Case<ExtractElementInst>(
                    [&](ExtractElementInst *extract) -> Value * {
                      return builder.CreateExtractElement(
                          getVal(extract->getOperand(0)),
                          getVal(extract->getOperand(1)));
                    })
                .Case<InsertElementInst>(
                    [&](InsertElementInst *insert) -> Value * {
                      return builder.CreateInsertElement(
                          getVal(insert->getOperand(0)),
                          getVal(insert->getOperand(1)),
                          getVal(insert->getOperand(2)));
                    })
                .Default([](Instruction *) -> Value * { return nullptr; });

        if (!newVal)
          continue;

        replacedValues[&inst] = newVal;
        SmallVector<User *, 8> users(inst.users());
        for (User *user : users)
          for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
            if (user->getOperand(i) == &inst)
              user->setOperand(i, newVal);
        toErase.push_back(&inst);
      }
    }

    for (Instruction *inst : toErase) {
      // Erase from map first to avoid stale-pointer reuse by LLVM's allocator.
      valueTypeMap.erase(inst);
      inst->eraseFromParent();
    }
  }

public:
  InterproceduralAddressSpacePropagationUpdater(
      llvm::Module &module, llvm::ModuleAnalysisManager &mam)
      : CallGraphUpdater(module, mam) {}

  /// Build the complete Value→Type map without touching IR.
  virtual bool analyze() final {
    llvm::LLVMContext &ctx = module.getContext();
    SmallVector<llvm::Value *> worklist;

    // Seed from kernel function arguments.
    for (llvm::Function &func : module) {
      if (func.isDeclaration() &&
          !func.getName().starts_with("air.atomic.global."))
        continue;
      if (!isKernelFunction(func))
        continue;

      functionsToUpdate.insert(&func);

      for (llvm::Argument &arg : func.args()) {
        llvm::Type *argTy = arg.getType();
        llvm::Type *newTy = nullptr;

        if (argTy->isPointerTy()) {
          if (cast<llvm::PointerType>(argTy)->getAddressSpace() ==
              MetalAddressSpace::None)
            newTy = llvm::PointerType::get(ctx, MetalAddressSpace::Device);
        } else if (argTy->isIntegerTy() || argTy->isFloatingPointTy()) {
          // Scalar args become constant buffer ptrs in the Metal signature;
          // record for the signature only (not the worklist). updateFunction
          // inserts a load so body uses see the original scalar type, avoiding
          // invalid mixed-type IR from propagating ptr addrspace(2).
          Attribute attr =
              func.getFnAttribute("metal.thread_index_in_simdgroup_idx");
          if (!attr.isValid() ||
              attr.getValueAsString() != std::to_string(arg.getArgNo()))
            valueTypeMap.try_emplace(
                &arg, llvm::PointerType::get(ctx, MetalAddressSpace::Constant));
          // Don't fall through to the generic recordNewType+worklist below.
        } else if (argTy->isArrayTy() || argTy->isStructTy()) {
          // Aggregate args become device ptrs in the Metal signature; propagate
          // the body type (struct with promoted nested ptrs) to arg uses.
          valueTypeMap.try_emplace(
              &arg, llvm::PointerType::get(ctx, MetalAddressSpace::Device));
          llvm::Type *bodyTy =
              createTypeWithAddressSpace(argTy, MetalAddressSpace::Device);
          if (bodyTy != argTy) {
            for (llvm::Use &use : arg.uses()) {
              auto *user = use.getUser();
              // computeNewResultType skips CallInst; propagate body type
              // directly into the callee parameter.
              if (auto *call = dyn_cast<CallInst>(user)) {
                llvm::Function *callee = call->getCalledFunction();
                if (callee && !callee->isDeclaration()) {
                  unsigned argNo = use.getOperandNo();
                  if (argNo < callee->arg_size()) {
                    if (!recordNewType(callee->getArg(argNo), bodyTy, worklist))
                      return false;
                    functionsToUpdate.insert(callee);
                  }
                }
              } else if (auto *inst = dyn_cast<Instruction>(user)) {
                if (llvm::Type *newInstTy =
                        computeNewResultType(inst, use.getOperandNo(), bodyTy))
                  if (newInstTy != inst->getType())
                    if (!recordNewType(inst, newInstTy, worklist))
                      return false;
              }
            }
          }
          // Don't fall through to the generic recordNewType below.
          newTy = nullptr;
        } else if (hasPointerWithoutAddressSpace(argTy)) {
          newTy = createTypeWithAddressSpace(argTy, MetalAddressSpace::Device);
        }

        if (newTy && !recordNewType(&arg, newTy, worklist))
          return false;
      }
    }

    // Promote any ptr addrspace(0) leaves in the loaded type to match.
    for (llvm::Function *func : functionsToUpdate) {
      if (func->isDeclaration())
        continue;
      for (BasicBlock &bb : *func) {
        for (Instruction &inst : bb) {
          auto *load = dyn_cast<LoadInst>(&inst);
          if (!load)
            continue;
          auto *ptrTy =
              dyn_cast<llvm::PointerType>(load->getPointerOperand()->getType());
          if (!ptrTy || ptrTy->getAddressSpace() == MetalAddressSpace::None)
            continue;
          MetalAddressSpace as =
              static_cast<MetalAddressSpace>(ptrTy->getAddressSpace());
          if (as != MetalAddressSpace::Device)
            continue;
          llvm::Type *newLoadTy =
              createTypeWithAddressSpace(load->getType(), as);
          if (newLoadTy != load->getType())
            if (!recordNewType(load, newLoadTy, worklist))
              return false;
        }
      }
    }

    if (!runWorklistPass(worklist))
      return false;

    // Handle the OptionalReg[T] type-pun pattern. Must run AFTER the main
    // worklist loop: callees are added to functionsToUpdate only when their
    // call sites are processed, so the type-pun alloca (which lives in the
    // inner helper, not the kernel entry) would be invisible if we scanned
    // before propagation.
    bool foundTypePun = false;
    if (!scanTypePunAllocas(worklist, foundTypePun))
      return false;

    if (foundTypePun && !runWorklistPass(worklist))
      return false;

    return !functionsToUpdate.empty();
  }

  /// Create new function with updated signature, move body, and
  /// rewrite instructions using the pre-built valueTypeMap.
  ///
  /// Rewrite happens *before* splice so valueTypeMap keys (computed against the
  /// old IR) stay valid. For each type-changing argument, a dummy instruction
  /// of the new type is inserted at the entry block top; after the rewritten
  /// body is spliced into newFunc, each dummy is replaced with the real arg.
  virtual llvm::Function *updateFunction(llvm::Function &func) final {
    llvm::LLVMContext &ctx = module.getContext();

    // Determine new parameter types from valueTypeMap.
    SmallVector<Type *, 8> newParamTypes;
    for (llvm::Argument &arg : func.args()) {
      llvm::Type *newTy = valueTypeMap.lookup(&arg);
      newParamTypes.push_back(newTy ? newTy : arg.getType());
    }

    llvm::Type *newRetTy = returnTypeMap.lookup(&func);
    FunctionType *newFuncTy = FunctionType::get(
        newRetTy ? newRetTy : func.getReturnType(), newParamTypes, false);

    KGEN_DEBUG(1, {
      llvm::dbgs() << KGEN_DEBUG_TYPE
                   << "addrspace propagation: updating function '"
                   << func.getName() << "'\n";
      llvm::dbgs() << KGEN_DEBUG_TYPE << ":   args:";
      for (llvm::Argument &arg : func.args()) {
        llvm::Type *newTy = valueTypeMap.lookup(&arg);
        llvm::dbgs() << " (" << *arg.getType() << " -> "
                     << *(newTy ? newTy : arg.getType()) << ")";
      }
      llvm::dbgs() << '\n';
      llvm::dbgs() << KGEN_DEBUG_TYPE << ":   ret: " << *func.getReturnType()
                   << " -> " << *(newRetTy ? newRetTy : func.getReturnType())
                   << '\n';
    });

    Function *newFunc = Function::Create(newFuncTy, func.getLinkage(),
                                         func.getName() + ".tmp", module);
    newFunc->copyAttributesFrom(&func);

    if (func.isDeclaration())
      return newFunc;

    // Insert dummy instructions for type-changing args, RAUW old args
    // with them so rewriteFunctionBody sees the right types. Post-splice, each
    // dummy is swapped for the real new argument and erased.
    SmallVector<std::pair<Instruction *, unsigned>, 8> argDummies;
    SmallVector<Instruction *, 4> extraToErase;
    SmallVector<unsigned, 8> sameTypeArgIndices;

    BasicBlock &entryBB = func.getEntryBlock();
    IRBuilder<> irBuilder(ctx);
    irBuilder.SetInsertPoint(entryBB.getFirstNonPHIIt());

    // RAUW without the type-equality assertion; needed when old and new types
    // differ (e.g., ptr → ptr addrspace(1)). Mismatches are resolved by
    // rewriteFunctionBody before the IR reaches any validating pass.
    auto replaceUsesUnsafe = [](Value *oldVal, Value *newVal) {
      SmallVector<Use *> uses;
      for (Use &u : oldVal->uses())
        uses.push_back(&u);
      for (Use *u : uses)
        u->set(newVal);
    };

    for (llvm::Argument &oldArg : func.args()) {
      unsigned argIdx = oldArg.getArgNo();
      llvm::Type *oldTy = oldArg.getType();
      llvm::Type *newTy = valueTypeMap.lookup(&oldArg);

      if (!newTy || newTy == oldTy) {
        // No type change: defer RAUW until after splice.
        sameTypeArgIndices.push_back(argIdx);
        continue;
      }

      if (oldTy->isPointerTy() && newTy->isPointerTy()) {
        // ptr → ptr addrspace(N): alloca+load dummy (never constant-folded).
        // Types differ, so use replaceUsesUnsafe.
        auto *dummyAlloca = cast<AllocaInst>(irBuilder.CreateAlloca(
            newTy, nullptr, oldArg.getName() + ".dummy.alloca"));
        auto *dummy = cast<LoadInst>(irBuilder.CreateLoad(
            newTy, dummyAlloca, oldArg.getName() + ".dummy"));
        replaceUsesUnsafe(&oldArg, dummy);
        argDummies.emplace_back(dummy, argIdx);
        extraToErase.push_back(dummyAlloca);
      } else if ((oldTy->isIntegerTy() || oldTy->isFloatingPointTy()) &&
                 newTy->isPointerTy()) {
        // Scalar → ptr: body sees the loaded scalar; dummy ptr → real arg
        // after splice. Same type, plain RAUW is safe.
        auto *dummyPtrAlloca = cast<AllocaInst>(irBuilder.CreateAlloca(
            newTy, nullptr, oldArg.getName() + ".dummy.ptr.alloca"));
        auto *dummyPtr = cast<LoadInst>(irBuilder.CreateLoad(
            newTy, dummyPtrAlloca, oldArg.getName() + ".dummy.ptr"));
        Value *loaded =
            irBuilder.CreateLoad(oldTy, dummyPtr, oldArg.getName() + ".loaded");
        oldArg.replaceAllUsesWith(loaded);
        argDummies.emplace_back(dummyPtr, argIdx);
        extraToErase.push_back(dummyPtrAlloca);
      } else if (isa<llvm::StructType, ArrayType>(oldTy) &&
                 newTy->isPointerTy()) {
        // Aggregate → ptr: load aggregate from dummy ptr (same type, plain
        // RAUW). Register the loaded value in valueTypeMap if its nested
        // pointers also need promotion.
        auto *dummyPtrAlloca = cast<AllocaInst>(irBuilder.CreateAlloca(
            newTy, nullptr, oldArg.getName() + ".dummy.ptr.alloca"));
        auto *dummyPtr = cast<LoadInst>(irBuilder.CreateLoad(
            newTy, dummyPtrAlloca, oldArg.getName() + ".dummy.ptr"));
        Value *loaded = irBuilder.CreateLoad(oldTy, dummyPtr,
                                             oldArg.getName() + ".agg.loaded");
        llvm::Type *loadTy =
            createTypeWithAddressSpace(oldTy, MetalAddressSpace::Device);
        if (loadTy != oldTy)
          valueTypeMap[loaded] = loadTy;
        oldArg.replaceAllUsesWith(loaded);
        argDummies.emplace_back(dummyPtr, argIdx);
        extraToErase.push_back(dummyPtrAlloca);
      } else {
        // Fallback: alloca+load dummy of the new type. Types differ, use
        // replaceUsesUnsafe.
        auto *dummyAlloca = irBuilder.CreateAlloca(
            newTy, nullptr, oldArg.getName() + ".dummy.alloca");
        Value *dummyLoad = irBuilder.CreateLoad(newTy, dummyAlloca,
                                                oldArg.getName() + ".dummy");
        replaceUsesUnsafe(&oldArg, dummyLoad);
        argDummies.emplace_back(cast<Instruction>(dummyLoad), argIdx);
        extraToErase.push_back(dummyAlloca);
      }
    }

    // Rewrite body in-place; valueTypeMap keys remain valid.
    rewriteFunctionBody(func);

    // Move rewritten body into the new function.
    newFunc->splice(newFunc->begin(), &func);

    // Wire up real arguments.
    auto oldArgIt = func.arg_begin();
    auto newArgIt = newFunc->arg_begin();
    for (size_t i = 0, e = func.arg_size(); i < e; ++i, ++oldArgIt, ++newArgIt)
      newArgIt->setName(oldArgIt->getName());

    for (unsigned idx : sameTypeArgIndices) {
      llvm::Argument *oldArg = func.getArg(idx);
      llvm::Argument *newArg = newFunc->getArg(idx);
      oldArg->replaceAllUsesWith(newArg);
    }

    for (auto [dummy, argIdx] : argDummies) {
      llvm::Argument *newArg = newFunc->getArg(argIdx);
      dummy->replaceAllUsesWith(newArg);
      dummy->eraseFromParent();
    }
    for (Instruction *extra : extraToErase)
      extra->eraseFromParent();

    if (isKernelFunction(*newFunc))
      addKernelPointerAttributes(*newFunc);

    return newFunc;
  }

  /// Create a new call to \p newFunc, inserting addrspacecast or bitcast
  /// instructions for any argument whose type differs from the new parameter
  /// type (e.g. [1 x i64] arg passed where i64 is expected after promotion).
  virtual llvm::Value *updateCall(llvm::CallInst &call, llvm::Function &newFunc,
                                  llvm::Function &callerFunc) final {
    llvm::FunctionType *newFuncTy = newFunc.getFunctionType();
    SmallVector<llvm::Value *> args;
    IRBuilder<> builder(&call);
    for (unsigned i = 0, e = call.arg_size(); i < e; ++i) {
      llvm::Value *arg = call.getArgOperand(i);
      if (i < newFuncTy->getNumParams()) {
        llvm::Type *expectedTy = newFuncTy->getParamType(i);
        if (arg->getType() != expectedTy) {
          if (arg->getType()->isPointerTy() && expectedTy->isPointerTy()) {
            arg = builder.CreateAddrSpaceCast(arg, expectedTy);
          } else if (CastInst::castIsValid(Instruction::BitCast, arg,
                                           expectedTy)) {
            arg = builder.CreateBitOrPointerCast(arg, expectedTy);
          } else if (auto *arrTy = dyn_cast<ArrayType>(arg->getType())) {
            // [1 x T] passed where T is expected after bitcast-wrapper strip.
            if (arrTy->getNumElements() == 1 &&
                arrTy->getElementType() == expectedTy) {
              arg = builder.CreateExtractValue(arg, {0});
            } else {
              llvm_unreachable(
                  "unhandled argument type mismatch in updateCall");
            }
          } else {
            llvm_unreachable("unhandled argument type mismatch in updateCall");
          }
        }
      }
      args.push_back(arg);
    }
    return CallInst::Create(&newFunc, args);
  }
};

} // anonymous namespace

/// Analyze call graph to find functions that require thread parameters
static bool addThreadParameters(llvm::Module &module,
                                llvm::ModuleAnalysisManager &mam) {

  AIRThreadParameterUpdater updater(module, mam);
  if (updater.analyze()) {
    updater.update();
    return true;
  }
  llvm_unreachable(
      "Failed to find kernel functions that require thread parameters.");
}

/// Analyze call graph to find functions that require address space propagation
/// For kernels that not only produce address space, but also rewrites functions
/// to accept arguments by pointer instead of by value
[[maybe_unused]] static bool
propagateAddressSpace(llvm::Module &module, llvm::ModuleAnalysisManager &mam) {
  AddressSpacePropagationUpdater updater(module, mam);
  if (updater.analyze()) {
    updater.update();
    return true;
  }
  llvm_unreachable("Failed to find kernel functions that require address space "
                   "propagation.");
}

/// New two-phase address space propagation. Use this instead of
/// propagateAddressSpace() once validation against the test suite passes.
static bool propagateIPAddressSpace(llvm::Module &module,
                                    llvm::ModuleAnalysisManager &mam) {
  InterproceduralAddressSpacePropagationUpdater updater(module, mam);
  if (updater.analyze()) {
    updater.update();
    return true;
  }
  llvm_unreachable("Failed to find kernel functions that require address space "
                   "propagation.");
}

/// Convert i64 operations to i32 for Metal compatibility
static bool convertI64OperationsToI32(Function &func) {
  LLVMContext &ctx = func.getContext();
  bool changed = false;

  // Collect all i64 instructions that need conversion
  SmallVector<Instruction *, 32> toConvert;
  SmallVector<Argument *, 4> i64Args;

  // Find i64 arguments (should already be converted by addThreadParameters)
  for (Argument &arg : func.args())
    if (arg.getType()->isIntegerTy(64))
      i64Args.push_back(&arg);

  // Find i64 instructions that derive from thread calculations
  for (BasicBlock &bb : func) {
    for (Instruction &inst : bb) {
      bool shouldConvert = false;

      if (inst.getType()->isIntegerTy(64)) {
        // Convert i64 operations to i32 for Metal compatibility
        if (isa<ZExtInst, SExtInst, BinaryOperator>(inst))
          shouldConvert = true;
      }

      // Special case for icmp: check operands rather than return type (which is
      // i1)
      if (auto *icmp = dyn_cast<ICmpInst>(&inst)) {
        if (icmp->getOperand(0)->getType()->isIntegerTy(64) ||
            icmp->getOperand(1)->getType()->isIntegerTy(64)) {
          shouldConvert = true;
        }
      }

      // Handle gep instructions with i64 indices separately
      if (auto *gep = dyn_cast<GetElementPtrInst>(&inst)) {
        for (unsigned i = 0, e = gep->getNumIndices(); i < e; ++i) {
          if (gep->getOperand(i + 1)->getType()->isIntegerTy(64)) {
            shouldConvert = true;
            break;
          }
        }
      }

      if (shouldConvert)
        toConvert.push_back(&inst);
    }
  }

  // Convert operations in multiple passes to handle dependencies correctly
  bool madeChanges = true;
  while (madeChanges && !toConvert.empty()) {
    madeChanges = false;
    SmallVector<Instruction *, 32> remainingToConvert;

    for (Instruction *inst : toConvert) {
      IRBuilder<> builder(inst);
      Value *replacement = nullptr;
      bool canConvertNow = true;

      // Check if all operands are ready (either i32 or can be converted)
      for (Value *operand : inst->operands()) {
        if (auto *operandInst = dyn_cast<Instruction>(operand)) {
          if (operandInst->getType()->isIntegerTy(64) &&
              std::find(toConvert.begin(), toConvert.end(), operandInst) !=
                  toConvert.end()) {
            canConvertNow = false; // Dependency not ready yet
            break;
          }
        }
      }

      if (!canConvertNow) {
        remainingToConvert.push_back(inst);
        continue;
      }

      if (auto *ze = dyn_cast<ZExtInst>(inst)) {
        // zext i32 %x to i64 -> keep as i32 (no zext needed)
        if (ze->getSrcTy()->isIntegerTy(32) &&
            ze->getDestTy()->isIntegerTy(64)) {
          replacement = ze->getOperand(0); // Just use the i32 value directly
        }
      } else if (auto *se = dyn_cast<SExtInst>(inst)) {
        // sext i32 %x to i64 -> keep as i32 (no sext needed)
        if (se->getSrcTy()->isIntegerTy(32) &&
            se->getDestTy()->isIntegerTy(64)) {
          replacement = se->getOperand(0); // Just use the i32 value directly
        }
      } else if (auto *binOp = dyn_cast<BinaryOperator>(inst)) {
        // Convert i64 arithmetic to i32
        if (binOp->getType()->isIntegerTy(64)) {
          Value *lhs = binOp->getOperand(0);
          Value *rhs = binOp->getOperand(1);

          // Truncate operands to i32 if they're i64 (skip if already i32)
          if (lhs->getType()->isIntegerTy(64))
            lhs = builder.CreateTrunc(lhs, Type::getInt32Ty(ctx));

          if (rhs->getType()->isIntegerTy(64))
            rhs = builder.CreateTrunc(rhs, Type::getInt32Ty(ctx));

          // Create new i32 operation
          replacement = builder.CreateBinOp(binOp->getOpcode(), lhs, rhs);
          if (binOp->hasNoSignedWrap())
            cast<BinaryOperator>(replacement)->setHasNoSignedWrap();
          if (binOp->hasNoUnsignedWrap())
            cast<BinaryOperator>(replacement)->setHasNoUnsignedWrap();
        }
      } else if (auto *icmp = dyn_cast<ICmpInst>(inst)) {
        // Convert i64 comparisons to i32
        Value *lhs = icmp->getOperand(0);
        Value *rhs = icmp->getOperand(1);

        // Truncate operands to i32 if they're i64 (skip if already i32)
        if (lhs->getType()->isIntegerTy(64))
          lhs = builder.CreateTrunc(lhs, Type::getInt32Ty(ctx));

        if (rhs->getType()->isIntegerTy(64))
          rhs = builder.CreateTrunc(rhs, Type::getInt32Ty(ctx));

        // Create new i32 comparison (result is still i1)
        replacement = builder.CreateICmp(icmp->getPredicate(), lhs, rhs);
      } else if (auto *gep = dyn_cast<GetElementPtrInst>(inst)) {
        // Convert gep with i64 indices to i32 indices
        SmallVector<Value *, 4> newIndices;
        bool needsConversion = false;

        for (unsigned i = 0, e = gep->getNumIndices(); i < e; ++i) {
          Value *idx = gep->getOperand(i + 1);
          if (idx->getType()->isIntegerTy(64)) {
            idx = builder.CreateTrunc(idx, Type::getInt32Ty(ctx));
            needsConversion = true;
          }
          newIndices.push_back(idx);
        }

        if (needsConversion) {
          replacement = builder.CreateGEP(gep->getSourceElementType(),
                                          gep->getPointerOperand(), newIndices);
          if (gep->isInBounds())
            cast<GetElementPtrInst>(replacement)->setIsInBounds(true);
        }
      }

      if (replacement && replacement != inst) {
        // Only replace if types match or if replacement can be safely used
        SmallVector<User *, 16> users(inst->users());
        for (User *user : users)
          for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
            if (user->getOperand(i) == inst)
              user->setOperand(i, replacement);
        inst->eraseFromParent();
        changed = true;
        madeChanges = true;
      } else {
        remainingToConvert.push_back(inst);
      }
    }

    toConvert = remainingToConvert;
  }

  return changed;
}

/// Replace AIR intrinsics with thread parameter access
static bool replaceAIRIntrinsics(Function &func) {
  // Check if function has thread parameters
  if (!func.hasFnAttribute("metal.thread_args_added"))
    return false;

  bool changed = false;

  // Find and replace all AIR intrinsic calls
  SmallVector<CallInst *, 32> intrinsicCalls;
  for (BasicBlock &bb : func) {
    for (Instruction &inst : bb) {
      if (auto *callInst = dyn_cast<CallInst>(&inst)) {
        if (Function *callee = callInst->getCalledFunction();
            callee && isAIRThreadIntrinsic(*callee)) {
          intrinsicCalls.push_back(callInst);
        }
      }
    }
  }

  const SmallVector<std::pair<std::string, llvm::Type *>> &metalArgs =
      getMetalArgsDescription(func.getContext());

  // Replace each intrinsic call with appropriate values
  for (CallInst *callInst : intrinsicCalls) {
    IRBuilder<> builder(callInst);
    StringRef intrinsicName = callInst->getCalledFunction()->getName();

    Value *result = nullptr;

    for (unsigned i = 0, e = metalArgs.size(); i < e; ++i) {
      if (!intrinsicName.starts_with("air." + metalArgs[i].first))
        continue;

      llvm::Type *type = metalArgs[i].second;
      result = func.getArg(func.arg_size() - e + i);

      // For the vector type generate extract element according to dimension
      // that is specified in the intrinsic name.
      if (isa<VectorType>(type)) {
        unsigned component = 0;
        if (intrinsicName.ends_with(".y"))
          component = 1;
        else if (intrinsicName.ends_with(".z"))
          component = 2;

        result = builder.CreateExtractElement(result, component);

        // For Metal compatibility, prefer i32 over i64 for thread indices
        // Metal AIR uses i32 for thread calculations
        if (callInst->getType() != result->getType()) {
          if (callInst->getType()->isIntegerTy(64) &&
              result->getType()->isIntegerTy(32)) {
            // Extend i32 to i64 if needed (for compatibility with existing
            // code)
            result = builder.CreateZExt(result, callInst->getType());
          } else if (result->getType()->isIntegerTy(64) &&
                     callInst->getType()->isIntegerTy(32)) {
            result = builder.CreateTrunc(result, callInst->getType());
          }
        }
      }
    }

    callInst->replaceAllUsesWith(result);
    callInst->eraseFromParent();
    changed = true;
  }

  return changed;
}

/// Update an instruction to use address space pointers instead of regular
/// pointers
static void updateInstructionForAddressSpace(User *user, Value *oldPtr,
                                             Value *newPtr) {
  if (auto *gep = dyn_cast<GetElementPtrInst>(user)) {
    // Update gep instruction to use address space pointer
    IRBuilder<> builder(gep);

    // Get gep indices
    SmallVector<Value *, 8> indices;
    for (auto &idx : gep->indices())
      indices.push_back(idx);

    // Create new gep with address space pointer - this will maintain the
    // address space
    Value *newGEP =
        builder.CreateGEP(gep->getSourceElementType(), newPtr, indices);

    // Now update all users of this gep to use the new address space version
    SmallVector<User *, 8> gepUsers(gep->users());
    for (User *gepUser : gepUsers)
      updateGEPUser(gepUser, gep, cast<GetElementPtrInst>(newGEP));

    gep->eraseFromParent();

  } else if (auto *phi = dyn_cast<PHINode>(user)) {
    // Handle PHI nodes by updating their incoming values
    for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e; ++i)
      if (phi->getIncomingValue(i) == oldPtr)
        phi->setIncomingValue(i, newPtr);
  } else {
    // For other instruction types, try direct replacement
    for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
      if (user->getOperand(i) == oldPtr)
        user->setOperand(i, newPtr);
  }
}

/// Update users of gep instructions to use address space correctly
static void updateGEPUser(User *user, GetElementPtrInst *oldGEP,
                          GetElementPtrInst *newGEP) {
  if (auto *load = dyn_cast<LoadInst>(user)) {
    // Replace load to use address space pointer directly
    IRBuilder<> builder(load);
    Value *newLoad = builder.CreateLoad(
        createTypeWithAddressSpace(load, newGEP->getType()), newGEP);

    // Copy alignment and other attributes
    if (auto *newLoadInst = dyn_cast<LoadInst>(newLoad)) {
      newLoadInst->setAlignment(load->getAlign());
      if (auto *tbaa = load->getMetadata("tbaa"))
        newLoadInst->setMetadata("tbaa", tbaa);
    }
    SmallVector<User *, 8> lusers(load->users());
    for (User *lu : lusers)
      updateInstructionForAddressSpace(lu, load, newLoad);
    load->eraseFromParent();
  } else if (auto *store = dyn_cast<StoreInst>(user)) {
    // Replace store to use address space pointer directly
    IRBuilder<> builder(store);
    auto *newStore = builder.CreateStore(store->getValueOperand(), newGEP);

    // Copy alignment and other attributes
    newStore->setAlignment(store->getAlign());
    if (auto *tbaa = store->getMetadata("tbaa"))
      newStore->setMetadata("tbaa", tbaa);
    store->eraseFromParent();
  } else {
    // For other instructions, replace the operand
    for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
      if (user->getOperand(i) == oldGEP)
        user->setOperand(i, newGEP);
  }
}

/// Eliminate addrspacecast instructions in a function
static bool eliminateAddrspaceCasts(Function &func) {
  bool changed = false;

  // Collect all addrspacecast instructions that need to be eliminated
  SmallVector<AddrSpaceCastInst *, 16> addrspaceCasts;
  for (BasicBlock &bb : func)
    for (Instruction &inst : bb)
      if (auto *cast = dyn_cast<AddrSpaceCastInst>(&inst))
        addrspaceCasts.push_back(cast);

  for (auto *cast : addrspaceCasts) {
    // Get the source and destination address spaces
    Value *srcPtr = cast->getOperand(0);
    Type *srcType = srcPtr->getType();
    Type *destType = cast->getType();

    // Check if this is a cast from addrspace(1) to addrspace(0)
    if (auto *srcPtrTy = dyn_cast<llvm::PointerType>(srcType)) {
      if (auto *destPtrTy = dyn_cast<llvm::PointerType>(destType)) {
        if (srcPtrTy->getAddressSpace() == MetalAddressSpace::Device &&
            destPtrTy->getAddressSpace() == 0) {
          // This is the problematic pattern: addrspacecast ptr addrspace(1) to
          // ptr We need to eliminate this and make all operations use
          // addrspace(1) directly

          // Replace all uses of the cast with operations that use addrspace(1)
          // directly
          SmallVector<User *, 16> users(cast->users());
          for (User *user : users) {
            if (auto *gep = dyn_cast<GetElementPtrInst>(user)) {
              // Replace gep with version that uses address space 1
              IRBuilder<> builder(gep);

              SmallVector<Value *, 8> indices;
              for (auto &idx : gep->indices())
                indices.push_back(idx);

              // Create new gep directly on the address space 1 pointer
              Value *newGEP = builder.CreateGEP(gep->getSourceElementType(),
                                                srcPtr, indices);

              // Update all users of this gep
              SmallVector<User *, 8> gepUsers(gep->users());
              for (User *gepUser : gepUsers) {
                if (auto *load = dyn_cast<LoadInst>(gepUser)) {
                  // Create new load that uses address space 1 directly
                  IRBuilder<> loadBuilder(load);
                  Value *newLoad =
                      loadBuilder.CreateLoad(load->getType(), newGEP);

                  // Copy attributes
                  if (auto *newLoadInst = dyn_cast<LoadInst>(newLoad)) {
                    newLoadInst->setAlignment(load->getAlign());
                    if (auto *tbaa = load->getMetadata("tbaa"))
                      newLoadInst->setMetadata("tbaa", tbaa);
                  }
                  load->replaceAllUsesWith(newLoad);
                  load->eraseFromParent();
                } else if (auto *store = dyn_cast<StoreInst>(gepUser)) {
                  // Create new store that uses address space 1 directly
                  IRBuilder<> storeBuilder(store);
                  auto *newStore = storeBuilder.CreateStore(
                      store->getValueOperand(), newGEP);

                  // Copy attributes
                  newStore->setAlignment(store->getAlign());
                  if (auto *tbaa = store->getMetadata("tbaa"))
                    newStore->setMetadata("tbaa", tbaa);
                  store->eraseFromParent();
                } else {
                  // For other users, try direct replacement
                  for (unsigned i = 0, e = gepUser->getNumOperands(); i < e;
                       ++i)
                    if (gepUser->getOperand(i) == gep)
                      gepUser->setOperand(i, newGEP);
                }
              }

              gep->eraseFromParent();

            } else {
              // For other users, replace with the original address space 1
              // pointer
              for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i)
                if (user->getOperand(i) == cast)
                  user->setOperand(i, srcPtr);
            }
          }
          cast->eraseFromParent();
          changed = true;
        } else if (srcPtrTy->getAddressSpace() ==
                   destPtrTy->getAddressSpace()) {
          // trivial case. Need to remove as LLVM does expect it's folded during
          // AddrSpaceCastInst's construction
          cast->replaceAllUsesWith(srcPtr);
          cast->eraseFromParent();
        }
      }
    }
  }

  return changed;
}

/// Gates `nosync`. Stricter than LangRef: monotonic atomics count too.
static bool synchronizesAcrossThreads(Function &f,
                                      const SetVector<Function *> &known) {
  for (Instruction &inst : instructions(f)) {
    if (inst.isAtomic() || inst.isVolatile())
      return true;
    auto *call = dyn_cast<CallBase>(&inst);
    if (!call)
      continue;
    if (call->isConvergent())
      return true;
    Function *callee = call->getCalledFunction();
    if (!callee)
      continue;
    // POP lowers Apple atomic RMW to an opaque call, past `isAtomic`.
    if (callee->getName().starts_with("air.atomic.") || known.contains(callee))
      return true;
  }
  return false;
}

/// Post-order visits callees first; an SCC shares one answer.
static SetVector<Function *>
findSynchronizingFunctions(Module &module, ModuleAnalysisManager &mam) {
  LazyCallGraph &graph = mam.getResult<LazyCallGraphAnalysis>(module);
  graph.buildRefSCCs();

  SetVector<Function *> synchronizing;
  for (LazyCallGraph::RefSCC &refSCC : graph.postorder_ref_sccs()) {
    for (LazyCallGraph::SCC &scc : refSCC) {
      bool any = false;
      for (LazyCallGraph::Node &node : scc)
        if (synchronizesAcrossThreads(node.getFunction(), synchronizing)) {
          any = true;
          break;
        }
      if (!any)
        continue;
      for (LazyCallGraph::Node &node : scc)
        synchronizing.insert(&node.getFunction());
    }
  }
  return synchronizing;
}

/// Fix function signatures and attributes for Metal compatibility
static bool fixMetalFunctionCompatibility(Module &module,
                                          ModuleAnalysisManager &mam) {
  const SetVector<Function *> synchronizing =
      findSynchronizingFunctions(module, mam);
  bool changed = false;
  LLVMContext &ctx = module.getContext();
  SmallVector<StringRef> attrsToStrip = {
      // This attribute does not make Metal crash as of now, but it makes
      // sense to remove it.
      "target-cpu",
      // Forwarding this attribute results in inability to create Metal
      // Compute Pipeline regardless if it's set on a kernel or function that
      // kernel calls.
      "target-features",
  };

  // Collect functions that need return type transformation
  SmallVector<Function *, 8> functionsToTransform;
  SmallVector<Function *, 8> functionsNeedingThreadIndexTransform;

  for (Function &f : module) {
    if (isKernelFunction(f)) {
      // Check if function needs return type transformation
      // Temporarily disable return type transformation to preserve function
      // body
      if (!f.getReturnType()->isVoidTy()) {
        // functionsToTransform.push_back(&f); // Disabled for now
      }

      // AGGRESSIVELY remove ALL attributes that cause xcrun metal to fail
      // We need to completely strip the function of problematic attributes

      // Build a completely clean attribute list
      AttributeList newAttrs;

      // Always set Metal-compatible calling convention
      if (f.getCallingConv() !=
          static_cast<CallingConv::ID>(MetalCallingConvention::C)) {
        f.setCallingConv(
            static_cast<CallingConv::ID>(MetalCallingConvention::C));
      }

      LLVMContext &ctx = f.getContext();
      // Create Metal-compatible attribute list based on working reference
      AttrBuilder metalAttrs(ctx);

      SmallVector<Attribute> attributes = {
          Attribute::get(ctx, Attribute::AttrKind::MustProgress),
          Attribute::get(ctx, Attribute::AttrKind::NoFree),
          Attribute::get(ctx, Attribute::AttrKind::NoRecurse),
          Attribute::get(ctx, Attribute::AttrKind::NoUnwind),
          Attribute::get(ctx, Attribute::AttrKind::WillReturn),
          Attribute::get(ctx, "approx-func-fp-math", "true"),
          Attribute::get(ctx, "frame-pointer", "all"),
          Attribute::get(ctx, "min-legal-vector-width", "0"),
          Attribute::get(ctx, "no-builtins"),
          Attribute::get(ctx, "no-infs-fp-math", "true"),
          Attribute::get(ctx, "no-nans-fp-math", "true"),
          Attribute::get(ctx, "no-signed-zeros-fp-math", "true"),
          Attribute::get(ctx, "no-trapping-math", "true"),
          Attribute::get(ctx, "stack-protector-buffer-size", "8"),
          Attribute::get(ctx, "unsafe-fp-math", "true"),
          Attribute::get(ctx, kMetalKernelAttr, "true"),
      };

      // Add the exact attributes that working Metal kernels have
      // (`nosync` is conditional; decided below).
      for (Attribute &attr : attributes)
        metalAttrs.addAttribute(attr);

      // NOTE: Do NOT add "air.kernel" as a function attribute
      // Metal kernels are identified through !air.kernel metadata, not function
      // attributes Adding it as a function attribute causes xcrun metallib to
      // fail

      // PRESERVE parameter attributes (like byval) while replacing function
      // attributes Get existing parameter attributes to preserve
      AttributeList oldAttrs = f.getAttributes();
      for (const Attribute &attr : oldAttrs.getFnAttrs())
        metalAttrs.addAttribute(attr);

      // After the merge, which would otherwise reinstate `nosync` (GPUA-130).
      if (synchronizing.contains(&f))
        metalAttrs.removeAttribute(Attribute::AttrKind::NoSync);
      else
        metalAttrs.addAttribute(Attribute::AttrKind::NoSync);

      // Start with Metal-compatible function attributes
      newAttrs = AttributeList::get(f.getContext(),
                                    AttributeList::FunctionIndex, metalAttrs);

      // Preserve parameter attributes from original function
      for (unsigned i = 0, e = f.arg_size(); i < e; ++i) {
        AttributeSet paramAttrs = oldAttrs.getParamAttrs(i);
        if (paramAttrs.hasAttributes()) {
          // Convert AttributeSet to AttrBuilder for addParamAttributes
          AttrBuilder paramBuilder(f.getContext());
          for (const Attribute &attr : paramAttrs)
            paramBuilder.addAttribute(attr);
          newAttrs =
              newAttrs.addParamAttributes(f.getContext(), i, paramBuilder);
        }
      }

      f.setAttributes(newAttrs);

      // Fix function linkage and attributes for AIR compatibility
      if (f.hasLocalLinkage()) {
        f.setLinkage(GlobalValue::InternalLinkage);
      } else {
        f.setLinkage(GlobalValue::ExternalLinkage);
      }
      f.setDSOLocal(false); // Remove dso_local for AIR compatibility
      f.setUnnamedAddr(
          GlobalValue::UnnamedAddr::Local); // Add local_unnamed_addr

      // Set Metal kernel calling convention
      f.setCallingConv(static_cast<CallingConv::ID>(MetalCallingConvention::C));

      changed = true;
    }
    for (auto attr : attrsToStrip)
      f.removeFnAttr(attr);
  }

  // Transform function return types to void
  for (Function *oldFunc : functionsToTransform) {
    // Create new function type with void return
    FunctionType *oldFT = oldFunc->getFunctionType();
    FunctionType *newFT = FunctionType::get(Type::getVoidTy(ctx),
                                            oldFT->params(), oldFT->isVarArg());

    // Store the original name before creating new function
    std::string originalName = oldFunc->getName().str();

    // Create new function with void return type
    Function *newFunc = Function::Create(newFT, oldFunc->getLinkage(),
                                         originalName + "_void_temp", &module);

    // Copy attributes (excluding return attributes)
    newFunc->setAttributes(
        oldFunc->getAttributes().removeParamAttributes(ctx, 0));

    // CRITICAL: Set calling convention to C for Metal compatibility
    newFunc->setCallingConv(CallingConv::C);

    // Copy function body
    if (!oldFunc->isDeclaration()) {
      // Move basic blocks to new function
      while (!oldFunc->empty()) {
        BasicBlock &bb = oldFunc->front();
        bb.removeFromParent();
        bb.insertInto(newFunc);
      }

      // Fix return instructions to return void
      for (BasicBlock &bb : *newFunc) {
        if (ReturnInst *ret = dyn_cast<ReturnInst>(bb.getTerminator())) {
          if (ret->getReturnValue()) {
            // Replace return with value with void return
            IRBuilder<> builder(ret);
            builder.CreateRetVoid();
            ret->eraseFromParent();
          }
        }
      }
    }

    // Replace all uses of old function with new function
    oldFunc->replaceAllUsesWith(newFunc);

    // Remove old function
    oldFunc->eraseFromParent();

    // Update name to original
    newFunc->setName(originalName);

    changed = true;
  }

  for (Function &f : module)
    changed |= replaceAIRIntrinsics(f);

  // Convert i64 operations to i32 for Metal compatibility
  // TEMPORARILY DISABLED: i64→i32 conversion causing invalid cast errors
  if (false)
    for (Function &f : module)
      changed |= convertI64OperationsToI32(f);

  // Eliminate addrspacecast instructions
  for (Function &f : module)
    changed |= eliminateAddrspaceCasts(f);

  return changed;
}

/// Final cleanup pass - aggressively strip all problematic attributes
static bool finalMetalCleanup(Module &module) {
  bool changed = false;

  for (Function &f : module) {
    if (isKernelFunction(f)) {
      // Ensure Metal-compatible attributes (final cleanup)
      // Don't change attributes here as they were already set properly
      // Just verify the function doesn't have problematic attributes
      if (f.hasFnAttribute("air.kernel"))
        f.removeFnAttr("air.kernel");

      changed = true;
    }
  }

  return changed;
}

/// Type info used for AIR metadata (informational; does not affect codegen).
struct MetalElemTypeInfo {
  unsigned size;
  unsigned align;
  StringRef name;
};

/// Return AIR element type info for a scalar LLVM type.
static MetalElemTypeInfo getMetalElemTypeInfo(Type *ty) {
  assert(!ty->isDoubleTy() && "'double' type is not supported by Apple GPU.");
  if (ty->isFloatTy())
    return {4, 4, "float"};
  if (ty->isHalfTy())
    return {2, 2, "half"};
  if (ty->isIntegerTy(8))
    return {1, 1, "char"};
  if (ty->isIntegerTy(16))
    return {2, 2, "short"};
  if (ty->isIntegerTy(32))
    return {4, 4, "int"};
  if (ty->isIntegerTy(64))
    return {8, 8, "long"};
  return {4, 4, "void"};
}

/// Add kernel attributes to GPU kernel functions
static bool addKernelAttributes(Module &module) {
  bool changed = false;
  LLVMContext &ctx = module.getContext();
  NamedMDNode *airKernelMD = module.getOrInsertNamedMetadata("air.kernel");

  for (Function &f : module) {
    // Check if this looks like a GPU kernel
    if (isKernelFunction(f)) {
      // Remove air.kernel attribute if present (Metal kernels don't use this)
      if (f.hasFnAttribute("air.kernel")) {
        f.removeFnAttr("air.kernel");
        changed = true;
      }

      // Check if kernel metadata already exists for this function
      bool hasKernelMetadata = false;
      for (unsigned i = 0, e = airKernelMD->getNumOperands(); i < e; ++i) {
        if (MDNode *kernelNode = airKernelMD->getOperand(i)) {
          if (kernelNode->getNumOperands() > 0) {
            if (auto *funcMD =
                    dyn_cast<ValueAsMetadata>(kernelNode->getOperand(0))) {
              if (funcMD->getValue() == &f) {
                hasKernelMetadata = true;
                break;
              }
            }
          }
        }
      }

      // Add kernel metadata if not already present
      if (!hasKernelMetadata) {
        // Create AIR-compatible kernel metadata with function reference
        SmallVector<Metadata *, 3> kernelMetadata;

        // Add function as metadata. Under LLVM's opaque-pointer model a
        // Function's type is already a plain `ptr`, so no explicit
        // function-pointer bitcast is needed for Metal AIR compatibility.
        kernelMetadata.push_back(llvm::ConstantAsMetadata::get(&f));

        // Add empty attributes metadata
        kernelMetadata.push_back(MDNode::get(ctx, ArrayRef<Metadata *>()));

        // Generate argument metadata for the kernel
        SmallVector<Metadata *, 8> argMetadata;
        unsigned argIndex = 0;
        for (const Argument &arg : f.args()) {
          SmallVector<Metadata *, 16> singleArgMD;

          // Argument index
          singleArgMD.push_back(ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(ctx), argIndex)));

          Type *argType = arg.getType();
          if (llvm::PointerType *ptrTy = dyn_cast<llvm::PointerType>(argType)) {
            // Device and constant buffers both use the "air.buffer" tag with
            // argIndex as their location_index. Dynamic/external Threadgroup
            // memory is handled via !air.global_bindings (not as a kernel
            // argument) and will never appear here.
            singleArgMD.push_back(MDString::get(ctx, "air.buffer"));
            singleArgMD.push_back(MDString::get(ctx, "air.location_index"));
            singleArgMD.push_back(ConstantAsMetadata::get(
                ConstantInt::get(Type::getInt32Ty(ctx), argIndex)));
            singleArgMD.push_back(ConstantAsMetadata::get(
                ConstantInt::get(Type::getInt32Ty(ctx), 1))); // Buffer type

            // Handle memory attributes

            // Check argument attributes for memory access patterns
            // By default use read_write access pattern
            std::string accessPattern = "air.read_write";

            unsigned argNo = arg.getArgNo();
            AttributeSet paramAttrs = f.getAttributes().getAttributes(
                AttributeList::FirstArgIndex + argNo);
            // FIXME: Revisit this code. There might be no need to check for
            // string representation and it might be better to use
            // f.hasParamAttribute() function directly.
            if (paramAttrs.hasAttributes()) {
              if (paramAttrs.hasAttribute(Attribute::ReadOnly) ||
                  paramAttrs.hasAttribute("readonly")) {
                // Check for readonly attribute (both enum and string forms)
                accessPattern = "air.read";
              } else if (paramAttrs.hasAttribute(Attribute::WriteOnly) ||
                         paramAttrs.hasAttribute("writeonly")) {
                // Check for writeonly attribute (both enum and string forms)
                accessPattern = "air.write";
              } else if (paramAttrs.hasAttribute("memory")) {
                // Check for memory attribute with specific patterns
                Attribute memAttr = paramAttrs.getAttribute("memory");
                if (memAttr.isStringAttribute()) {
                  StringRef memValue = memAttr.getValueAsString();
                  if (memValue == "argmem: readonly")
                    accessPattern = "air.read";
                  else if (memValue == "argmem: writeonly")
                    accessPattern = "air.write";
                  else if (memValue == "argmem: readwrite")
                    accessPattern = "air.read_write";
                }
              } else if (arg.hasByValAttr() || arg.hasByRefAttr()) {
                // Metal cannot take arguments neither byval, nor byref.
                llvm_unreachable(
                    "byval attribute is not supported for Metal target");
              }
            }

            // Constant memory (address space 2) is always read-only
            if (ptrTy->getAddressSpace() == 2) {
              accessPattern = "air.read";
            }

            singleArgMD.push_back(MDString::get(ctx, accessPattern));

            singleArgMD.push_back(MDString::get(ctx, "air.address_space"));
            singleArgMD.push_back(ConstantAsMetadata::get(ConstantInt::get(
                Type::getInt32Ty(ctx), ptrTy->getAddressSpace())));

            // Add type information - get dynamic type from pointer analysis
            singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));

            // Build pointer map to get dynamic type information
            auto pointerMap = PointerRewriter::buildPointerMap(module);
            auto it = pointerMap.find(&arg);

            std::string typeStr = "void"; // Default fallback

            if (it != pointerMap.end() && it->second) {
              // Use the typed pointer information from pointer analysis
              auto *typedPtr = it->second;
              auto *elemType = typedPtr->getElementType();

              typeStr = getMetalElemTypeInfo(elemType).name;
            }

            singleArgMD.push_back(MDString::get(ctx, typeStr));

            if (!arg.getName().empty()) {
              singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
              singleArgMD.push_back(MDString::get(ctx, arg.getName()));
            }
          } else if (argType->isPointerTy() &&
                     argType->getPointerAddressSpace() ==
                         MetalAddressSpace::Constant) {
            // Constant buffer parameter (converted from scalar)
            singleArgMD.push_back(MDString::get(ctx, "air.address_space"));
            singleArgMD.push_back(ConstantAsMetadata::get(ConstantInt::get(
                Type::getInt32Ty(ctx), 2))); // addrspace(2) for constant memory

            singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
            singleArgMD.push_back(MDString::get(
                ctx, "constant int*")); // Constant buffer reference

            if (!arg.getName().empty()) {
              singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
              singleArgMD.push_back(MDString::get(ctx, arg.getName()));
            }
          } else if (argType->isIntegerTy()) {
            StringRef argName = arg.getName();
            if (f.hasFnAttribute("metal.thread_args_added") &&
                argName == "thread_index_in_simdgroup") {
              singleArgMD.push_back(
                  MDString::get(ctx, "air.thread_index_in_simdgroup"));
              singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
              singleArgMD.push_back(MDString::get(ctx, "uint"));
            } else {
              // This case should not occur anymore since scalars are converted
              // to constant buffers But keep as fallback
              singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
              singleArgMD.push_back(MDString::get(ctx, "int"));
            }

            if (!arg.getName().empty()) {
              singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
              singleArgMD.push_back(MDString::get(ctx, arg.getName()));
            }
          } else if (auto *vecTy = dyn_cast<VectorType>(argType)) {
            // Check if this is one of our thread parameter vectors
            StringRef argName = arg.getName();
            if (f.hasFnAttribute("metal.thread_args_added") &&
                vecTy->getElementType()->isIntegerTy(32) &&
                vecTy->getElementCount().getFixedValue() == 3 &&
                (argName == "threadgroup_position_in_grid" ||
                 argName == "thread_position_in_threadgroup" ||
                 argName == "threads_per_threadgroup" ||
                 argName == "threads_per_grid")) {

              // Add the proper AIR thread attribute
              if (argName == "threadgroup_position_in_grid") {
                singleArgMD.push_back(
                    MDString::get(ctx, "air.threadgroup_position_in_grid"));
              } else if (argName == "thread_position_in_threadgroup") {
                singleArgMD.push_back(
                    MDString::get(ctx, "air.thread_position_in_threadgroup"));
              } else if (argName == "threads_per_threadgroup") {
                singleArgMD.push_back(
                    MDString::get(ctx, "air.threads_per_threadgroup"));
              } else if (argName == "threads_per_grid") {
                singleArgMD.push_back(
                    MDString::get(ctx, "air.threads_per_grid"));
              }

              singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
              singleArgMD.push_back(MDString::get(ctx, "uint3"));

              if (!arg.getName().empty()) {
                singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
                singleArgMD.push_back(MDString::get(ctx, arg.getName()));
              }

              // Add air.arg_unused attribute like native Metal
              singleArgMD.push_back(MDString::get(ctx, "air.arg_unused"));

            } else {
              // Generic vector types
              singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
              singleArgMD.push_back(MDString::get(ctx, "vector"));

              if (!arg.getName().empty()) {
                singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
                singleArgMD.push_back(MDString::get(ctx, arg.getName()));
              }
            }
          } else {
            // Generic argument
            singleArgMD.push_back(MDString::get(ctx, "air.arg_type_name"));
            singleArgMD.push_back(MDString::get(ctx, "unknown"));

            if (!arg.getName().empty()) {
              singleArgMD.push_back(MDString::get(ctx, "air.arg_name"));
              singleArgMD.push_back(MDString::get(ctx, arg.getName()));
            }
          }

          argMetadata.push_back(MDNode::get(ctx, singleArgMD));
          ++argIndex;
        }

        kernelMetadata.push_back(MDNode::get(ctx, argMetadata));

        if (f.hasFnAttribute("air.max_work_group_size")) {
          StringRef maxWorkGroupSizeStr =
              f.getFnAttribute("air.max_work_group_size").getValueAsString();
          uint32_t maxWorkGroupSize = std::stoi(maxWorkGroupSizeStr.str());

          kernelMetadata.push_back(MDNode::get(
              ctx, {MDString::get(ctx, "air.max_work_group_size"),
                    ConstantAsMetadata::get(ConstantInt::get(
                        Type::getInt32Ty(ctx), maxWorkGroupSize))}));
        }

        MDNode *kernelNode = MDNode::get(ctx, kernelMetadata);
        airKernelMD->addOperand(kernelNode);

        changed = true;
      }
    }
  }

  return changed;
}

/// Convert extern addrspace(3) globals to global_bindings
static bool promoteThreadgroupGlobalsToGlobalBindings(Module &module) {
  LLVMContext &ctx = module.getContext();

  // Collect external globals residing in ThreadGroup addrspace(3).
  SmallVector<GlobalVariable *, 4> tgGlobals;
  for (GlobalVariable &gv : module.globals()) {
    if (gv.getAddressSpace() == MetalAddressSpace::ThreadGroup &&
        gv.hasExternalLinkage() && gv.isDeclaration())
      tgGlobals.push_back(&gv);
  }
  if (tgGlobals.empty())
    return false;

  // Type for the value held in the addrspace(2) constant: a pointer into
  // threadgroup memory.
  Type *tgPtrTy = llvm::PointerType::get(ctx, MetalAddressSpace::ThreadGroup);

  SmallVector<Metadata *, 4> bindingEntries;

  auto makeI32MD = [&](int32_t v) -> Metadata * {
    return ConstantAsMetadata::get(
        ConstantInt::getSigned(Type::getInt32Ty(ctx), v));
  };
  auto makeStrMD = [&](StringRef s) -> Metadata * {
    return MDString::get(ctx, s);
  };

  for (GlobalVariable *oldGV : tgGlobals) {
    // Infer element type name and byte size for AIR metadata (best-effort;
    // these are informational for debuggers / profilers and do not affect
    // correctness).
    Type *elemTy = oldGV->getValueType();
    MetalElemTypeInfo elemInfo = getMetalElemTypeInfo(elemTy);
    unsigned elemSize = elemInfo.size;
    unsigned elemAlign = elemInfo.align;
    StringRef elemName = elemInfo.name;

    // Save the name before we steal it for the new global.
    std::string globalName = oldGV->getName().str();

    // Create the replacement:
    //   @name = internal unnamed_addr addrspace(2) externally_initialized
    //           constant ptr addrspace(3) undef, align 8
    auto *newGV = new GlobalVariable(
        module,
        tgPtrTy, // value type: ptr addrspace(3)
        /*isConstant=*/true, GlobalValue::InternalLinkage,
        UndefValue::get(tgPtrTy), // initializer: undef (runtime fills)
        "",                       // name assigned via takeName below
        /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
        MetalAddressSpace::Constant, // addrspace(2)
        /*isExternallyInitialized=*/true);
    newGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    newGV->setAlignment(Align(8));
    newGV->takeName(oldGV);

    // Replace every use of the old addrspace(3) global with a load from the
    // new addrspace(2) constant.  Collect the use-list first to avoid
    // invalidation while we iterate.
    SmallVector<Use *, 8> uses;
    for (Use &u : oldGV->uses())
      uses.push_back(&u);

    for (Use *u : uses) {
      // Each use is expected to be a direct instruction operand — the
      // extern_ptr_symbol always materialises at the external_memory()
      // call site in Mojo-generated IR.
      auto *i = cast<Instruction>(u->getUser());
      IRBuilder<> builder(i);
      Value *tgPtr = builder.CreateLoad(tgPtrTy, newGV, globalName + ".tg");
      u->set(tgPtr);
    }

    assert(oldGV->use_empty() && "unexpected non-instruction user of "
                                 "addrspace(3) threadgroup global");
    oldGV->eraseFromParent();

    // Build the !air.global_binding metadata entry.
    //
    // Format (matching Apple's metalfe output):
    //   !{!"air.global_binding", ptr addrspace(2) @gv, !details}
    //   !details = !{i32 -1, !"air.buffer",
    //                !"air.location_index", i32 N, i32 1,
    //                !"air.read_write",
    //                !"air.address_space", i32 3,
    //                !"air.arg_type_size", i32 S,
    //                !"air.arg_type_align_size", i32 A,
    //                !"air.arg_type_name", !"T",
    //                !"air.arg_name", !"name"}
    //
    // The leading i32 -1 signals "not a kernel function argument" to the
    // AIR linker / Metal framework.

    // All globals bind to slot 0, matching CUDA's single-base-address model.
    // See the function-level comment for the multi-symbol rationale.
    MDNode *detailNode =
        MDNode::get(ctx, {
                             makeI32MD(-1),
                             makeStrMD("air.buffer"),
                             makeStrMD("air.location_index"),
                             makeI32MD(0),
                             makeI32MD(1),
                             makeStrMD("air.read_write"),
                             makeStrMD("air.address_space"),
                             makeI32MD(MetalAddressSpace::ThreadGroup),
                             makeStrMD("air.arg_type_size"),
                             makeI32MD(elemSize),
                             makeStrMD("air.arg_type_align_size"),
                             makeI32MD(elemAlign),
                             makeStrMD("air.arg_type_name"),
                             makeStrMD(elemName),
                             makeStrMD("air.arg_name"),
                             makeStrMD(globalName),
                         });

    bindingEntries.push_back(
        MDNode::get(ctx, {
                             makeStrMD("air.global_binding"),
                             ConstantAsMetadata::get(newGV),
                             detailNode,
                         }));
  }

  // Attach all binding entries to the module-level !air.global_bindings node.
  NamedMDNode *globalBindingsMD =
      module.getOrInsertNamedMetadata("air.global_bindings");
  for (Metadata *entry : bindingEntries)
    globalBindingsMD->addOperand(cast<MDNode>(entry));

  return true;
}

/// Perform essential transformations for Metal compatibility of the module. The
/// function is used by new and legacy passes
static bool transformModuleForMetal(Module &module, ModuleAnalysisManager &mam,
                                    const MetalVersionInfo &version) {
  bool changed = false;

  // NOTE: Pointer rewriting is now deferred to the bitcode writer stage
  // to avoid redundant expensive analysis passes during compilation.
  // BitcodeWriter17 will handle all pointer transformations.

  // Transform the module for AIR compatibility
  changed |= configureModuleForMetal(module);
  changed |= transformTargetTriple(module, version);
  // Check pointer compatibility for debugging
  changed |= checkMetalPointerCompatibility(module);
  changed |= addAIRMetadata(module, version);
  // Convert extern threadgroup globals → constant global bindings
  changed |= promoteThreadgroupGlobalsToGlobalBindings(module);
  changed |= addThreadParameters(module, mam);
  changed |= propagateIPAddressSpace(module, mam);
  changed |= fixMetalFunctionCompatibility(module, mam);
  changed |= addKernelAttributes(module);
  // Final aggressive cleanup
  changed |= finalMetalCleanup(module);

  // Restore AIR target triple and data layout for final output
  if (changed) {
    // Restore AIR target triple from metadata
    if (auto *airTargetFlag = module.getModuleFlag("metal.air_target_triple")) {
      if (auto *airTargetMD = dyn_cast<llvm::MDString>(airTargetFlag)) {
        std::string airTriple = airTargetMD->getString().str();
        module.setTargetTriple(llvm::Triple(airTriple));
      }
    }

    // Restore Metal data layout from metadata
    if (auto *airDataLayoutFlag =
            module.getModuleFlag("metal.air_data_layout")) {
      if (auto *airDataLayoutMD = dyn_cast<llvm::MDString>(airDataLayoutFlag)) {
        std::string metalDataLayout = airDataLayoutMD->getString().str();
        module.setDataLayout(metalDataLayout);
      }
    }

    // Remove internal metadata flags that shouldn't be in final output
    if (auto *moduleFlagsMetadata = module.getModuleFlagsMetadata()) {
      llvm::SmallVector<llvm::MDNode *, 8> flagsToKeep;
      bool foundInternalMetadata = false;

      for (unsigned i = 0, e = moduleFlagsMetadata->getNumOperands(); i < e;
           ++i) {
        if (auto *flag = moduleFlagsMetadata->getOperand(i)) {
          if (flag->getNumOperands() >= 2) {
            if (auto *flagName =
                    dyn_cast<llvm::MDString>(flag->getOperand(1))) {
              StringRef name = flagName->getString();
              if (name == "metal.air_target_triple" ||
                  name == "metal.air_data_layout") {
                // Remove internal metadata used during compilation
                foundInternalMetadata = true;
                // Skip this flag
                continue;
              }
            }
          }
          flagsToKeep.push_back(flag);
        }
      }

      // Rebuild module flags without internal metadata
      if (foundInternalMetadata) {
        moduleFlagsMetadata->eraseFromParent();
        NamedMDNode *newFlags =
            module.getOrInsertNamedMetadata("llvm.module.flags");
        for (auto *flag : flagsToKeep)
          newFlags->addOperand(flag);
      }
    }
  }
  return changed;
}

} // anonymous namespace

namespace M::KGEN {

/// Clean up internal Metal attributes that were added during transformation
/// but shouldn't be visible in the final Metal AIR output
static void cleanupInternalMetalAttributes(Module &module) {
  const SmallVector<std::pair<std::string, llvm::Type *>> &metalArgs =
      getMetalArgsDescription(module.getContext());
  // Remove internal function attributes
  for (Function &func : module) {
    // Remove thread-parameter tracking attributes.
    for (unsigned i = 0, e = metalArgs.size(); i < e; ++i) {
      std::string attrName = "metal." + metalArgs[i].first;
      if (func.hasFnAttribute(attrName))
        func.removeFnAttr(attrName);
    }
  }
}

/// Remove freeze instructions for LLVM 5.0 compatibility
static bool removeUnsupportedInstructions(llvm::Module &module) {
  // Find freeze instructions
  llvm::SmallVector<llvm::Instruction *, 8> worklist;
  llvm::SmallVector<llvm::Function *, 8> funcToRemove;
  for (llvm::Function &func : module) {
    for (llvm::BasicBlock &bb : func) {
      for (llvm::Instruction &inst : bb) {
        if (isa<llvm::FreezeInst>(&inst))
          worklist.push_back(&inst);
        if (auto *intr = dyn_cast<llvm::IntrinsicInst>(&inst)) {
          if (intr->getIntrinsicID() == llvm::Intrinsic::lifetime_end ||
              intr->getIntrinsicID() == llvm::Intrinsic::lifetime_start) {
            worklist.push_back(&inst);
          }
        }
      }
    }
    if (func.isDeclaration() && func.isIntrinsic()) {
      if (func.getIntrinsicID() == llvm::Intrinsic::lifetime_end ||
          func.getIntrinsicID() == llvm::Intrinsic::lifetime_start) {
        funcToRemove.push_back(&func);
      }
    }
  }

  if (worklist.empty() && funcToRemove.empty())
    return false;

  // Replace freeze instructions by their operand
  for (llvm::Instruction *inst : worklist) {
    if (auto *freezeInst = dyn_cast<llvm::FreezeInst>(inst))
      freezeInst->replaceAllUsesWith(freezeInst->getOperand(0));
    inst->eraseFromParent();
  }
  for (llvm::Function *func : funcToRemove) {
    func->eraseFromParent();
  }
  return true;
}

/// Replace fneg instructions with fsub for LLVM 5.0 compatibility
static bool replaceFNeg(llvm::Module &module) {
  // Find fneg instructions
  llvm::SmallVector<llvm::UnaryOperator *, 8> worklist;
  for (llvm::Function &func : module)
    for (llvm::BasicBlock &bb : func)
      for (llvm::Instruction &inst : bb)
        if (auto *op = llvm::dyn_cast<llvm::UnaryOperator>(&inst))
          if (op->getOpcode() == llvm::Instruction::FNeg)
            worklist.push_back(op);

  if (worklist.empty())
    return false;

  // Replace fneg instructions by fsub instructions
  llvm::IRBuilder<> builder(module.getContext());
  for (llvm::UnaryOperator *op : worklist) {
    builder.SetInsertPoint(op);
    llvm::Value *in = op->getOperand(0);
    llvm::Value *zero = llvm::ConstantFP::get(in->getType(), -0.0);
    op->replaceAllUsesWith(builder.CreateFSub(zero, in));
    op->eraseFromParent();
  }
  return true;
}

/// Convert newer LLVM attributes to LLVM 14 compatible equivalents
/// for better compatibility with older Metal toolchains
static void downgradeLLVMFnAttributesForCompatibility(Module &module) {
  LLVMContext &ctx = module.getContext();

  // Process all functions (including declarations like llvm.smax.i32)
  for (Function &f : module) {
    AttributeList attrs = f.getAttributes();
    AttributeList newAttrs = attrs;
    bool functionChanged = false;

    // Process each attribute index (function attributes, return attributes, and
    // parameter attributes)
    for (unsigned i = AttributeList::FunctionIndex,
                  e = attrs.getNumAttrSets() + AttributeList::FirstArgIndex;
         i < e; ++i) {
      AttributeSet attrSet = attrs.getAttributes(i);
      if (!attrSet.hasAttributes())
        continue;

      AttrBuilder ab(ctx, attrSet);
      bool setChanged = false;

      // Handle memory attributes
      if (ab.contains("memory")) {
        llvm::Attribute memAttr = attrSet.getAttribute("memory");
        if (memAttr.isStringAttribute()) {
          StringRef memValue = memAttr.getValueAsString();
          ab.removeAttribute("memory");

          if (memValue == "none") {
            // Convert memory(none) to readnone
            ab.addAttribute(llvm::Attribute::ReadNone);
          } else if (memValue == "argmem: readwrite") {
            // For argmem: readwrite, we'll use the string attribute for
            // compatibility
            ab.addAttribute("argmemonly");
          } else if (memValue == "argmem: readonly") {
            // For argmem: readonly, use argmemonly + readonly
            ab.addAttribute("argmemonly");
            ab.addAttribute(llvm::Attribute::ReadOnly);
          } else if (memValue == "argmem: writeonly") {
            // For argmem: writeonly, use argmemonly + writeonly
            ab.addAttribute("argmemonly");
            ab.addAttribute(llvm::Attribute::WriteOnly);
          }
          // For any other memory attributes, just remove them
          setChanged = true;
        }
      }

      // Remove noundef attribute if present
      if (ab.contains("noundef")) {
        ab.removeAttribute("noundef");
        setChanged = true;
      }

      if (setChanged) {
        newAttrs = newAttrs.removeAttributesAtIndex(ctx, i);
        newAttrs = newAttrs.addAttributesAtIndex(ctx, i, ab);
        functionChanged = true;
      }
    }

    if (functionChanged)
      f.setAttributes(newAttrs);
  }
}

// Implementation of MetalAIRPass::run
PreservedAnalyses MetalAIRPass::run(Module &module,
                                    ModuleAnalysisManager &mam) {
  // Only run on Metal GPU targets
  if (!isMetalTarget(module))
    llvm_unreachable("Only run on Metal GPU targets");

  // Derive Metal/AIR/macOS version from the "target-features" attribute
  llvm::StringRef featuresFromIR;
  for (const Function &f : module) {
    if (f.hasFnAttribute("target-features")) {
      featuresFromIR = f.getFnAttribute("target-features").getValueAsString();
      break;
    }
  }
  MetalVersionInfo version = getMetalVersionFromFeatures(featuresFromIR);

  // Clean up internal Metal attributes and downgrade LLVM features before final
  // output
  if (transformModuleForMetal(module, mam, version)) {
    cleanupInternalMetalAttributes(module);
    downgradeLLVMFnAttributesForCompatibility(module);

    // Remove LLVM 5.0 incompatible instructions
    removeUnsupportedInstructions(module);
    replaceFNeg(module);
  }

  return PreservedAnalyses::all();
}

/// Register the pass with the new pass manager
void registerMetalAIRPass(PassBuilder &pb) {
  pb.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "metal-air") {
          MPM.addPass(MetalAIRPass());
          return true;
        }
        return false;
      });
}

} // namespace M::KGEN
