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
// Packages an AIR bitcode module into a .metallib container without the
// Metal toolchain, so Metal kernels can be built on hosts without Xcode.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_COMPILER_TARGET_METAL_METALLIBWRITER_H
#define MOJO_COMPILER_TARGET_METAL_METALLIBWRITER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Wraps `airBitcode`, the AIR serialization of `module`, in a metallib
/// container that `MTLDevice newLibraryWithData:` accepts. Every kernel listed
/// in the module's `!air.kernel` metadata becomes a library function backed by
/// the same bitcode. No reflection section is written: the runtime derives
/// pipeline argument reflection from the AIR metadata itself.
llvm::Expected<std::string> writeMetallib(const llvm::Module &module,
                                          llvm::StringRef airBitcode);

} // namespace M::KGEN

#endif // MOJO_COMPILER_TARGET_METAL_METALLIBWRITER_H
