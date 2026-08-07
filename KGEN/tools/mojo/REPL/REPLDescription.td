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

include "DescriptionBase.td"

def REPLDescription : MojoCommandDescription<
  "repl", "Launches the Mojo REPL.",
  "Launches a Mojo read-evaluate-print loop (REPL) environment, which provides "
  "interactive development in the terminal. You can also start the REPL by "
  "simply running `mojo`.\n"
  "\n"
  "Any number of options and arguments may be specified on the command line. "
  "These are then forwarded to the underlying lldb tool, which runs the REPL.",
  [UsageInfo<"lldb-options", "", /*variadicInput=*/false>]>, Index<3>;
