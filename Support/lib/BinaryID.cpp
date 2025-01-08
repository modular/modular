//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BinaryID.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

#if defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <stdint.h>
#else
#include <elf.h>

extern char __build_id_start;
extern char __build_id_end;
#endif // __APPLE__

using namespace M;

std::string M::getBinaryID() {
  std::string str;
  llvm::raw_string_ostream os(str);

#if defined(__APPLE__)
  // NOTE: This code can run in a dylib or executable, we need to fetch the
  // address of whichever one it comes from
  Dl_info info;
  if (dladdr((void *)&getBinaryID, &info) == 0)
    assert(false && "dladdr failed");
  auto *execHeader = (const struct mach_header_64 *)info.dli_fbase;

  // Get the header of the current binary or shared library, and find the UUID
  // load command
  uintptr_t command = (uintptr_t)execHeader + sizeof(struct mach_header_64);
  for (uint32_t idx = 0; idx < execHeader->ncmds; ++idx) {
    if (((const struct load_command *)command)->cmd == LC_UUID) {
      const struct uuid_command *cmd = (const struct uuid_command *)command;
      for (unsigned char i : cmd->uuid)
        os << llvm::format("%02x", i);
      break;
    } else {
      command += ((const struct load_command *)command)->cmdsize;
    }
  }
#else
  Elf64_Nhdr *hdr = (Elf64_Nhdr *)&__build_id_start;
  assert(hdr->n_type == NT_GNU_BUILD_ID && "invalid section type");

  const char *s = s = &__build_id_start + sizeof(Elf64_Nhdr) + hdr->n_namesz;
  for (; s < &__build_id_end; ++s)
    os << llvm::format("%02hhx", *s);
#endif // __APPLE__

  return str;
}
