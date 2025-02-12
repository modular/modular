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

defm target_accelerator : CommandOptionHidden<["--", "-"], "target-accelerator",
  TargetOptionGroup, Index<7>, MetaVarName<"ACCELERATOR">,
  HelpText<"Sets the accelerator used for computation.">>;

defm mcmodel: CommandOptionHidden<["--", "-"], "mcmodel",
  TargetOptionGroup, Index<8>, MetaVarName<"MCMODEL">,
  HelpText<"Sets the CodeModel for object file, i.e. small (default), medium or large.">>;

defm large_data_threshold: CommandOptionHidden<["--", "-"], "large-data-threshold",
  TargetOptionGroup, Index<9>, MetaVarName<"LDT">,
  HelpText<"Choose large data threshold for x86_64 medium code model.">>;

defm loop_unrolling_warn_threshold: CommandOptionHidden<["--", "-"], "loop-unrolling-warn-threshold",
  TargetOptionGroup, Index<10>, MetaVarName<"LUWT">,
  HelpText<"Sets loop unrolling factor warning threshold, default is 1024, set to 0 to disable the warning.">>;



#endif // KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
