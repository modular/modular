//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DOC_H
#define MOJO_DOC_H

namespace M {

class SubcommandRegistry;

/// Initializes the `doc` subcommand and its various options, and registers
/// its callback function with the registry.
void registerDocSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_DOC_H
