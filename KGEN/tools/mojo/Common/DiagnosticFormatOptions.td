//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a diagnostic format option as part of the "common options"
// group, such that it appears in the same group as `--help`.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD

include "Support/Driver/CommonOptions.td"

defm diagnostic_format : CommandOption<["--", "-"], "diagnostic-format",
  CommonOptionGroup, Index<0>, MetaVarName<"FORMAT">,
  HelpText<"The format in which diagnostics and error messages are printed. "
           "Must be one of \"text\" or \"json\" (\"text\" is the default).">>;

#endif // KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD
