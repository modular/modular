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
//
// This file defines the option that turns the parameter verifier on and off,
// for the subcommands that run the compiler's pass pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_RUNPARAMETERVERIFIEROPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_RUNPARAMETERVERIFIEROPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"

def run_parameter_verifier : Joined<["--", "-"], "run-parameter-verifier=">,
  Group<CompilationOptionGroup>, Flags<[HelpHidden]>, MetaVarName<"BOOL">,
  HelpText<
    "Whether to run the parameter verifier, which checks the parameters in "
    "the compiler's intermediate representation: `true` or `false`. The "
    "checks find compiler bugs, not errors in Mojo code. THIS IS A DEVELOPER OPTION.">,
  MarshallingInfoCustom<CompilationOpts<"runParameterVerifier">,
    [{makeChoiceNormalizer<bool>({{"true", true}, {"false", false}})}]>;

#endif // KGEN_TOOLS_MOJO_COMMON_RUNPARAMETERVERIFIEROPTIONS_TD
