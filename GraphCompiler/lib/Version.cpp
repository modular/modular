//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Version.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Config/Version.h"
#include "GraphCompiler/MGPDialect/RT/BinaryID.h"
#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "Mojo/CompilerRT/BinaryID.h"
#include "Support/BinaryID.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

#include "xxh3.h"
#include "xxhash.h"

std::string M::MO::getGraphCompilerVersionString() {
  // TODO(GEX-1565): Also include binary IDs of other shared libraries (e.g.
  // MOGGLoader) whose updates can change the compiled MEF.
  return M::getModularVersionString() + M::getBinaryID() + "-" +
         M::MGPRT::getMGPRTBinaryID() + "-" +
         M::Driver::getMojoBindingsBinaryID() + "-" +
         M::AsyncRT::getRuntimeGlobalsBinaryID() + "-" +
         M::KGEN::getCompilerRTBinaryID() + "-" +
         M::getModularVersion().buildType;
}

std::string M::MO::getMOGGCompilerVersionString() {
  static std::string version = []() {
    std::string versionString = M::getModularVersionString() +
                                M::getBinaryID() + "-" +
                                M::KGEN::getCompilerRTBinaryID() + "-" +
                                M::getModularVersion().buildType;
    return llvm::utohexstr(llvm::xxh3_64bits(versionString),
                           /*LowerCase=*/true,
                           /*Width=*/16);
  }();
  return version;
}
