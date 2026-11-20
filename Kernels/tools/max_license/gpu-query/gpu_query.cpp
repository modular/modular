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

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/StringExtras.h"

#include <cstdlib>

using namespace M;
using Attribute = M::Driver::Device::Attribute;

using llvm::StringRef;

namespace {

enum SubCommand {
  kDefaultCommand,
  kApi,
  kArch,
  kArchName,
  kTargetAccelerator,
  kVendor,
};

struct GpuQueryCLIOptions {
  M::cl::opt<SubCommand> gpuQuerySubCommand{
      M::cl::desc("Choose gpu-query command:"),
      M::cl::values(
          clEnumValN(kApi, "api", "API used by the GPU"),
          clEnumValN(kArch, "arch", "architecture of the GPU"),
          clEnumValN(kVendor, "vendor", "vendor of the GPU"),
          clEnumValN(kArchName, "arch-name", "architecture name of the GPU"),
          clEnumValN(
              kTargetAccelerator, "target-accelerator",
              "value to pass as --target-accelerator to \"mojo build\""))};
};

// This is a tiny implementation of struct gpu.host.Info
struct GpuInfo {
  GpuInfo(StringRef name, StringRef vendor, StringRef version)
      : name(name), vendor(vendor), version(version) {}
  StringRef name;
  StringRef vendor;
  StringRef version;
};

} // namespace

static GpuInfo mapToGpuInfo(const std::string &acceleratorArch) {
  if (acceleratorArch == "nvidia:80")
    return GpuInfo("A100", "nvidia_gpu", "sm_80");
  if (acceleratorArch == "nvidia:86")
    return GpuInfo("A10", "nvidia_gpu", "sm_86");
  if (acceleratorArch == "nvidia:87")
    return GpuInfo("OrinNano", "nvidia_gpu", "sm_87");
  if (acceleratorArch == "nvidia:89")
    return GpuInfo("L4", "nvidia_gpu", "sm_89");
  if (acceleratorArch == "nvidia:90")
    return GpuInfo("H100", "nvidia_gpu", "sm_90a");
  if (acceleratorArch == "amdgpu:94")
    return GpuInfo("MI300X", "amd_gpu", "CDNA3");

  // Unknown architecture.
  return GpuInfo("UNKNOWN", "UNKNOWN", "UNKNOWN");
}

template <typename T>
static std::string errOrString(ErrorOr<T> val) {
  return val.isError() ? val.getError() : std::to_string(val.takeValue());
}

#if !MODULAR_HAL_DEVICE_CONTEXT
template <>
std::string errOrString(ErrorOr<M::Driver::DriverVersion &> val) {
  return val.isError() ? val.getError() : val.get().toString();
}
#endif // !MODULAR_HAL_DEVICE_CONTEXT

static std::string printDefaultInfo(const M::Driver::DeviceContext &ctx) {
  std::stringstream stream;
  // TODO: Fill in the remaining information being printed by the mojo
  // version.
  const M::Driver::Device &device = ctx.device();

  auto memoryTotalString = [](ErrorOr<M::Driver::Device::MemInfo> val) {
    return !val.isError() ? humanMemorySize((*val).total) : val.getError();
  };
  auto memoryIntString = [](ErrorOr<int> val) {
    return !val.isError() ? humanMemorySize(*val) : val.getError();
  };
  auto attrString = [&](M::Driver::Device::Attribute attr) {
    return errOrString(device.getAttribute(attr));
  };

  stream << "name: " << device.name() << "\n";

#if !MODULAR_HAL_DEVICE_CONTEXT
  stream << "driver_version: " << errOrString(device.getDriverVersion())
         << "\n";

  stream << "api_version: " << errOrString(device.getApiVersion()) << "\n";
#endif // !MODULAR_HAL_DEVICE_CONTEXT

  stream << "memory: " << memoryTotalString(ctx.getMemoryInfo()) << "\n";

  stream << "compute_capability: "
         << attrString(Attribute::COMPUTE_CAPABILITY_MAJOR) << "."
         << attrString(Attribute::COMPUTE_CAPABILITY_MINOR) << "\n";

  stream << "clock_rate: " << attrString(Attribute::CLOCK_RATE) << "\n";

  stream << "warp_size: " << attrString(Attribute::WARP_SIZE) << "\n";

  stream << "max_threads_per_block: "
         << attrString(Attribute::MAX_THREADS_PER_BLOCK) << "\n";

  stream << "max_shared_memory_per_block: "
         << memoryIntString(
                device.getAttribute(Attribute::MAX_SHARED_MEMORY_PER_BLOCK))
         << "\n";

  stream << "max_block: (x=" << attrString(Attribute::MAX_BLOCK_DIM_X)
         << ", y=" << attrString(Attribute::MAX_BLOCK_DIM_Y)
         << ", z=" << attrString(Attribute::MAX_BLOCK_DIM_Z) << ")\n";

  stream << "max_grid: (x=" << attrString(Attribute::MAX_GRID_DIM_X)
         << ", y=" << attrString(Attribute::MAX_GRID_DIM_Y)
         << ", z=" << attrString(Attribute::MAX_GRID_DIM_Z) << ")\n";

  stream << "max_access_window_size: "
         << attrString(Attribute::MAX_ACCESS_POLICY_WINDOW_SIZE) << "\n";

  stream << "SM count: " << attrString(Attribute::MULTIPROCESSOR_COUNT) << "\n";

  stream << "Max threads per SM: "
         << attrString(Attribute::MAX_THREADS_PER_MULTIPROCESSOR) << "\n";

  stream << "Max registers per block: "
         << attrString(Attribute::MAX_REGISTERS_PER_BLOCK) << "\n";

  stream << "Max registers per multiprocessor: "
         << attrString(Attribute::MAX_REGISTERS_PER_MULTIPROCESSOR) << "\n";

  stream << "Max shared memory per SM: "
         << memoryIntString(device.getAttribute(
                Attribute::MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
         << "\n";

  return stream.str();
}

int main(int argc, char **argv) {
  GpuQueryCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Build Info Tool");

  ErrorOr<M::Driver::DeviceContextRef> ctxOr =
      M::Driver::DeviceContext::create("gpu");
  if (ctxOr.isError()) {
    // If we do not have a GPU installed, then we print the info below and exit
    // with exit code 2.
    llvm::outs() << ctxOr.getError() << "\n";
    llvm::outs() << "GPU is not present\n";
    return 2;
  }

  // A GPU is available, so we can make calls without checking for its presence.
  std::string acceleratorArch = M::Driver::Device::getAcceleratorArchOrEmpty();
  GpuInfo gpuInfo = mapToGpuInfo(acceleratorArch);
  M::Driver::DeviceContextRef ctx = ctxOr.takeValue();

  switch (cli.gpuQuerySubCommand) {
  case kApi:
    llvm::outs() << ctx->device().api() << "\n";
    break;
  case kArch:
    llvm::outs() << gpuInfo.version << "\n";
    break;
  case kArchName:
    llvm::outs() << gpuInfo.name << "\n";
    break;
  case kTargetAccelerator:
    llvm::outs() << M::Driver::Device::getAcceleratorArchOrEmpty() << "\n";
    break;
  case kVendor:
    llvm::outs() << gpuInfo.vendor << "\n";
    break;
  case kDefaultCommand: {
    // By default print information about the GPU.
    llvm::outs() << printDefaultInfo(*ctx.getPointer()) << "\n";
    break;
  }
  }
  return EXIT_SUCCESS;
}
