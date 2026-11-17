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
// This file defines the option for the namespace that a tool precompiles a
// package to be imported under. It lists among the including tool's output
// options, so the tool defines its `OutputOptionGroup` before including it.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_NAMESPACEOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_NAMESPACEOPTIONS_TD

include "Common/Marshalling.td"

// The `SmallVector<std::string> outputNamespace` being built.
def OutputNamespace : KeyPathAndMacro<"", "outputNamespace", "MOJO_">;

defm namespace : CommandOption<["--", "-"], "namespace", OutputOptionGroup,
  MetaVarName<"NAMESPACE">, HelpText<
    "Sets the dotted namespace the package is imported under. A package named "
    "`pkg` built with `--namespace foo.bar` must be placed at "
    "`foo/bar/pkg.mojoc` under an import root and imported as `foo.bar.pkg`; "
    "it can't be imported under any other name."
  >>,
  MarshallingInfoCustom<OutputNamespace, "normalizeOutputNamespace">;

#endif // KGEN_TOOLS_MOJO_COMMON_NAMESPACEOPTIONS_TD
