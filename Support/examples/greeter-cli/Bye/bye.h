//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EXAMPLES_GREETER_CLI_BYE_H
#define SUPPORT_EXAMPLES_GREETER_CLI_BYE_H

namespace M {

class SubcommandRegistry;

// This adds a `bye` subcommand function to the registry, under the name "bye".
void registerByeSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // SUPPORT_EXAMPLES_GREETER_CLI_BYE_H
