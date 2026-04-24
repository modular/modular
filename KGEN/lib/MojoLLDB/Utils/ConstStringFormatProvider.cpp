//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Provides a local definition of
// `llvm::format_provider<lldb_private::ConstString>::format` so that
// `libMojoLLDB.dylib` can resolve references emitted by unoptimised builds
// (e.g. `--config=debug-modular`) without statically linking
// `@llvm-project//lldb:Utility`. Pulling that dep in would drag
// `lldb:Host.o` into the plugin's static link set, giving it a duplicate
// copy of LLDB's `HostInfoBase` (including a file-scope `g_fields` pointer
// that is never initialised), which under macOS's two-level namespace
// crashes `mojo repl` at plugin load time. Matches the upstream
// implementation in `lldb/source/Utility/ConstString.cpp`. See MOTO-1573.

#include "lldb/Utility/ConstString.h"
#include "llvm/Support/FormatProviders.h"

void llvm::format_provider<lldb_private::ConstString>::format(
    const lldb_private::ConstString &CS, llvm::raw_ostream &OS,
    llvm::StringRef Options) {
  format_provider<StringRef>::format(CS.GetStringRef(), OS, Options);
}
