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
// This file defines the target queries, which print information about the
// targets and exit. Only the tools that answer them include it.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TARGETQUERYOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_TARGETQUERYOPTIONS_TD

include "Common/TargetOptions.td"

//===----------------------------------------------------------------------===//
// Print Target Information Options
//===----------------------------------------------------------------------===//

def print_effective_target : Flag<["--"], "print-effective-target">,
  Group<TargetOptionGroup>, Index<12>,
  HelpText<"Print the effective target configuration after absorbing all "
           "command-line flags and exit.">,
  MarshallingInfoFlag<DriverOpts<"printEffectiveTarget">>;

def print_supported_targets : Flag<["--"], "print-supported-targets">,
  Group<TargetOptionGroup>, Index<13>,
  HelpText<"Print all available target names and exit.">,
  MarshallingInfoFlag<DriverOpts<"printSupportedTargets">>;

def print_supported_cpus : Flag<["--"], "print-supported-cpus">,
  Group<TargetOptionGroup>, Index<14>,
  HelpText<"Print valid CPU names for the specified target and exit. "
           "Requires --target-triple.">,
  MarshallingInfoFlag<DriverOpts<"printSupportedCpus">>;

def print_supported_accelerators : Flag<["--"], "print-supported-accelerators">,
  Group<TargetOptionGroup>, Index<15>,
  HelpText<"Print all supported GPU and accelerator architectures and exit.">,
  MarshallingInfoFlag<DriverOpts<"printSupportedAccelerators">>;

#endif // KGEN_TOOLS_MOJO_COMMON_TARGETQUERYOPTIONS_TD
