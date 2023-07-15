//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_FORMAT_H
#define MOJO_FORMAT_H

namespace M {

class SubcommandRegistry;

/// Initializes the `format` subcommand and its various options, and registers
/// its callback function with the registry.
void registerFormatSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_FORMAT_H
