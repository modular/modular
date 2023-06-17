//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "llvm/ADT/StringRef.h"

#if defined(__x86_64__) && defined(__linux__)
#include "Support/SymbolExport.h"
#include <asm/prctl.h>   /* Definition of ARCH_* constants */
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

enum class XFeature : size_t {
  kXTileCfg = 17,
  kXTileData = 18,
  kMask_XTileCfg = (1 << kXTileCfg),
  kMask_XTileData = (1 << kXTileData),
  kMask_Xtile = (kMask_XTileCfg | kMask_XTileData)
};

// This function must be called before using Intel AMX
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT bool
KGEN_CompilerRT_Init_Intel_AMX() {
  unsigned long bitmask = 0;
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask))
    return false;
  if (bitmask & static_cast<unsigned long>(XFeature::kMask_XTileData))
    return true;

  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFeature::kXTileData))
    return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                  // allowed

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) ||
      !(bitmask & static_cast<unsigned long>(XFeature::kMask_XTileData)))
    return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
  return true;
}
#endif

/// Register the intel AMX functions.
void M::KGEN::registerIntelAMX(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
#if defined(__x86_64__) && defined(__linux__)
  funcs.push_back({"KGEN_CompilerRT_Init_Intel_AMX",
                   (void *)&KGEN_CompilerRT_Init_Intel_AMX});
#endif
}
