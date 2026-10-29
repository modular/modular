//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD

include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def TargetOptionGroup : OptionGroup<"Target options">, Index<6>;

defm target_triple : CommandOption<["--", "-"], "target-triple",
  TargetOptionGroup, MetaVarName<"TRIPLE">,
  HelpText<"Sets the compilation target triple. Defaults to the host target.">>,
  MarshallingInfoString<TargetArg<"triple">>;

defm target_cpu : CommandOption<["--", "-"], "target-cpu",
  TargetOptionGroup, MetaVarName<"CPU">,
  HelpText<"Sets the compilation target CPU. Defaults to the host CPU.">>,
  MarshallingInfoString<TargetArg<"cpu">>;

defm target_features : CommandOption<["--", "-"], "target-features",
  TargetOptionGroup, MetaVarName<"FEATURES">,
  HelpText<"Sets the compilation target CPU features. Defaults to the host "
           "features.">>,
  MarshallingInfoString<TargetArg<"features">>;

defm target_abi : CommandOption<["--", "-"], "target-abi",
  TargetOptionGroup, MetaVarName<"ABI">,
  HelpText<"Sets the target ABI name (e.g. `lp64d`), recorded as a "
           "`target-abi` LLVM module flag. Unset by default.">>,
  MarshallingInfoString<TargetArg<"abi">>;

defm march : CommandOption<["--", "-"], "march", TargetOptionGroup,
  MetaVarName<"ARCHITECTURE">,
  HelpText<"Sets the architecture for which to generate code.">>,
  MarshallingInfoString<TargetArg<"march">>;

defm mcpu : CommandOption<["--", "-"], "mcpu", TargetOptionGroup,
  MetaVarName<"CPU">, HelpText<"Sets the CPU for which to generate code.">>,
  MarshallingInfoString<TargetArg<"mcpu">>;

defm mtune : CommandOption<["--", "-"], "mtune", TargetOptionGroup,
  MetaVarName<"TUNE">,
  HelpText<"Sets the CPU for which to tune code.">>,
  MarshallingInfoString<TargetArg<"mtune">>;

defm target_accelerator : CommandOption<["--", "-"], "target-accelerator",
  TargetOptionGroup, MetaVarName<"ACCELERATOR">,
  HelpText<"Sets the GPU or accelerator architecture for heterogeneous "
           "computing (e.g., sm_90 for NVIDIA H100, gfx942 for AMD MI300).">>,
  MarshallingInfoString<TargetArg<"accelerator">>;

defm mcmodel: CommandOptionHidden<["--", "-"], "mcmodel",
  TargetOptionGroup, MetaVarName<"MCMODEL">,
  HelpText<"Sets the CodeModel for object file, i.e. small (default), medium or large.">>,
  MarshallingInfoString<TargetArg<"mcmodel">>;

defm large_data_threshold: CommandOptionHidden<["--", "-"], "large-data-threshold",
  TargetOptionGroup, MetaVarName<"LDT">,
  HelpText<"Choose large data threshold for x86_64 medium code model.">>,
  MarshallingInfoString<TargetArg<"largeDataThreshold">>;

defm relocation_model: CommandOptionHidden<["--", "-"], "relocation-model",
  TargetOptionGroup, MetaVarName<"MODEL">,
  HelpText<"Sets the relocation model. Supported values: `static`, `pic` "
           "(default), `dynamic-no-pic`, `ropi`, `rwpi`, `ropi-rwpi`.">>,
  MarshallingInfoString<TargetArg<"relocationModel">>;

#endif // KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
