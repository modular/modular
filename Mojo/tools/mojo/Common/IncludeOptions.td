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
// This file defines the compilation option group and the include path option,
// for subcommands that parse Mojo source without compiling it.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_INCLUDEOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_INCLUDEOPTIONS_TD

include "Support/Driver/DriverCommand.td"

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<5>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<3>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

#endif // KGEN_TOOLS_MOJO_COMMON_INCLUDEOPTIONS_TD
