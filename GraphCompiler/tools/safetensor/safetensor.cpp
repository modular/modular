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
//
// This is a command line utility that reads a safetensor file and prints the
// metadata and tensor info to the console.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

using namespace M;

namespace {

/// CLI options for safetensor.
class SafeTensorCLOptions : public CommonCLOptions {
public:
  class Options : public CommonOptions {};

  Options &options;
  using CommonCLOptions::CommonCLOptions;

  SafeTensorCLOptions(int argc, char **argv, Options &opts,
                      bool skipInitLLVM = true)
      : CommonCLOptions(argc, argv, opts, skipInitLLVM), options(opts),
        SafeTensorOptionsCategory("SafeTensor command line options") {}

  llvm::cl::OptionCategory SafeTensorOptionsCategory;

  llvm::cl::opt<bool> showMetadata{
      "metadata", cl::desc("Display the metadata in the safetensor file"),
      llvm::cl::cat(SafeTensorOptionsCategory)};

  llvm::cl::opt<std::string> tensorInfo{
      "tensor-info", cl::desc("Display the tensor-info in the safetensor file"),
      llvm::cl::cat(SafeTensorOptionsCategory)};
};
} // namespace

static int reportError(const Twine &errorMessage) {
  llvm::errs() << "safetensor: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  SafeTensorCLOptions::Options opts;
  SafeTensorCLOptions cli(argc, argv, opts);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "safetensor command line options\n");

  if (opts.inputFilename == "-")
    return reportError("stdin not supported, please specify an input file");

  auto device = M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
  if (device.isError())
    return reportError(device.getError());
  ErrorOr<SafeTensor> safeTensorOr =
      SafeTensor::get(opts.inputFilename, std::move(*device));
  if (safeTensorOr.isError())
    return reportError(safeTensorOr.getError());

  const auto &safeTensor = safeTensorOr.get();

  if (cli.showMetadata) {
    llvm::outs() << safeTensor.getMetadata() << "\n";
  } else if (!cli.tensorInfo.empty()) {
    ErrorOr<SafeTensor::TensorInfo> info =
        safeTensor.getTensorInfo(cli.tensorInfo);
    if (info.isError())
      return reportError(info.getError());

    llvm::outs() << info->toJSON() << "\n";
  } else {
    for (const auto &[key, value] : safeTensor.getTensorInfo()) {
      llvm::outs() << "tensor name: " << key << '\n';
      llvm::outs() << "value: " << value.toJSON() << '\n';
    }
  }

  return EXIT_SUCCESS;
}
