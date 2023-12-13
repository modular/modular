//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EXAMPLES_GREETER_CLI_HI_H
#define SUPPORT_EXAMPLES_GREETER_CLI_HI_H

namespace M {

class SubcommandRegistry;

// This adds a `hi` subcommand function to the registry, under the name "hi".
void registerHiSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // SUPPORT_EXAMPLES_GREETER_CLI_HI_H
