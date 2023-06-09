//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DOC_H
#define MOJO_DOC_H

#include "Support/CommandLine.h"
#include "mojo-driver.h"
#include "llvm/Support/CommandLine.h"

namespace M {

class SubCommandRegistry;

/// Initializes the `doc` subcommand and its various options, and registers
/// its callback function with the registry.
void registerDocSubCommand(SubCommandRegistry &registry);

} // namespace M

#endif // MOJO_DOC_H
