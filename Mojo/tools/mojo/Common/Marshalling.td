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
// expanding its `MOJO_OPTION_WITH_MARSHALLING` rows. The fields' C++
// initializers are the options' defaults, so the default values given to the
// `MarshallingInfo` classes are not read.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD
#define KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD

include "llvm/Option/OptParser.td"

// A field of the `KGEN::FrontendOptions frontendOptions` being built.
class FrontendOpts<string member>
    : KeyPathAndMacro<"frontendOptions.", member, "MOJO_">;

// A field of the `KGEN::CompilationOptions compilationOptions` being built.
class CompilationOpts<string member>
    : KeyPathAndMacro<"compilationOptions.", member, "MOJO_">;

// A field of the `TargetArgs targetArgs` being built.
class TargetArg<string member>
    : KeyPathAndMacro<"targetArgs.", member, "MOJO_">;

// A field of the `DriverOptions driverOptions` being built.
class DriverOpts<string member>
    : KeyPathAndMacro<"driverOptions.", member, "MOJO_">;

// A field of the driver's `State state`, which reports the driver's own
// errors and warnings.
class DriverState<string member>
    : KeyPathAndMacro<"state.", member, "MOJO_">;

// An option that a Mojo-specific normalizer parses, for values that need
// validation or conversion to a C++ type.
class MarshallingInfoCustom<KeyPathAndMacro kpm, code normalizer>
    : MarshallingInfo<kpm, "{}">, Normalizer<normalizer>;

#endif // KGEN_TOOLS_MOJO_COMMON_MARSHALLING_TD
