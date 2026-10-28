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
// This file defines the key paths that marshalled options store into. Each key
// path names a field of a local variable that the subcommand declares before
// expanding the matching `<PREFIX>OPTION_WITH_MARSHALLING` rows.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD
#define KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD

include "llvm/Option/OptParser.td"

// A field of the `KGEN::FrontendOptions frontendOptions` being built, expanded
// through `FRONTEND_OPTION_WITH_MARSHALLING`.
class FrontendOpts<string member>
    : KeyPathAndMacro<"frontendOptions.", member, "FRONTEND_">;

#endif // KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD
