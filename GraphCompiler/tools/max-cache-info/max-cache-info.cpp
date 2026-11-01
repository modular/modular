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

#include "GraphCompiler/Driver/Compile/CompileDriver.h"
#include "Support/CommandLine.h"

using namespace M;

namespace {

struct CacheInfoCLIOptions {
  cl::opt<bool> printCacheDir{"print-cache-dir",
                              cl::desc("Print MAX cache directory")};
  cl::opt<bool> printWeightsCacheDir{
      "print-weights-cache-dir", cl::desc("Print MAX weights cache directory")};
};

} // namespace

int main(int argc, char **argv) {
  CacheInfoCLIOptions cli;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MAX Cache Info Tool");

  bool any = false;
  bool ok = true;

  if (cli.printCacheDir) {
    any = true;
    if (auto cacheDir = getMaxCacheDir()) {
      llvm::outs() << *cacheDir << '\n';
    } else {
      ok = false;
      llvm::errs() << "unable to find MAX cache dir\n";
    }
  }
  if (cli.printWeightsCacheDir) {
    any = true;
    if (auto weightsCacheDir = getWeightsCacheDir()) {
      llvm::outs() << *weightsCacheDir << '\n';
    } else {
      ok = false;
      llvm::errs() << "unable to find MAX weights cache dir\n";
    }
  }

  if (!any) {
    llvm::errs()
        << "no options specified; see --help for available info requests\n";
    ok = false;
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
