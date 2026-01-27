//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def TargetOptionGroup : OptionGroup<"Target options">, Index<6>;

defm target_triple : CommandOption<["--", "-"], "target-triple",
  TargetOptionGroup, Index<1>, MetaVarName<"TRIPLE">,
  HelpText<"Sets the compilation target triple. Defaults to the host target.">>;

defm target_cpu : CommandOption<["--", "-"], "target-cpu",
  TargetOptionGroup, Index<2>, MetaVarName<"CPU">,
  HelpText<"Sets the compilation target CPU. Defaults to the host CPU.">>;

defm target_features : CommandOption<["--", "-"], "target-features",
  TargetOptionGroup, Index<3>, MetaVarName<"FEATURES">,
  HelpText<"Sets the compilation target CPU features. Defaults to the host "
           "features.">>;

defm march : CommandOption<["--", "-"], "march", TargetOptionGroup,
  Index<4>, MetaVarName<"ARCHITECTURE">,
  HelpText<"Sets the architecture for which to generate code.">>;

defm mcpu : CommandOption<["--", "-"], "mcpu", TargetOptionGroup, Index<5>,
  MetaVarName<"CPU">, HelpText<"Sets the CPU for which to generate code.">>;

defm mtune : CommandOption<["--", "-"], "mtune", TargetOptionGroup,
  Index<6>, MetaVarName<"TUNE">,
  HelpText<"Sets the CPU for which to tune code.">>;

defm target_accelerator : CommandOption<["--", "-"], "target-accelerator",
  TargetOptionGroup, Index<7>, MetaVarName<"ACCELERATOR">,
  HelpText<"Sets the GPU or accelerator architecture for heterogeneous "
           "computing (e.g., sm_90 for NVIDIA H100, gfx942 for AMD MI300).">>;

defm mcmodel: CommandOptionHidden<["--", "-"], "mcmodel",
  TargetOptionGroup, Index<8>, MetaVarName<"MCMODEL">,
  HelpText<"Sets the CodeModel for object file, i.e. small (default), medium or large.">>;

defm large_data_threshold: CommandOptionHidden<["--", "-"], "large-data-threshold",
  TargetOptionGroup, Index<9>, MetaVarName<"LDT">,
  HelpText<"Choose large data threshold for x86_64 medium code model.">>;

//===----------------------------------------------------------------------===//
// Print Target Information Options
//===----------------------------------------------------------------------===//

def print_effective_target : Flag<["--"], "print-effective-target">,
  Group<TargetOptionGroup>, Index<10>,
  HelpText<"Print the effective target configuration after absorbing all "
           "command-line flags and exit.">;

def print_supported_targets : Flag<["--"], "print-supported-targets">,
  Group<TargetOptionGroup>, Index<11>,
  HelpText<"Print all available target names and exit.">;

def print_supported_cpus : Flag<["--"], "print-supported-cpus">,
  Group<TargetOptionGroup>, Index<12>,
  HelpText<"Print valid CPU names for the specified target and exit. "
           "Requires --target-triple.">;

#endif // KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
