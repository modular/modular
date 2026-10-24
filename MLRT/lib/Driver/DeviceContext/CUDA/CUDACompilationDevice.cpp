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
// This file implements the CUDA compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#include "CUDACompilationDevice.h"

#include "CUDACompilationDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "NVML.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace llvm;

namespace M::Driver {

CUDACompilationDevice::CUDACompilationDevice(std::string targetArch)
    : CompilationDevice(Device::cudaAPI, std::move(targetArch)) {}

CUDACompilationDevice::~CUDACompilationDevice() = default;

SmallVector<std::string>
CUDACompilationDevice::setupCompilationOptionsNVPTXCompiler(
    StringRef debugLevel, int32_t optimizationLevel, StringRef gpuName) {
  // TODO: add more options needed.
  // https://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compilation-options

  static constexpr int kDefaultOptimizationLevel = 4;

  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  SmallVector<std::string, 3> options;
  options.push_back("--gpu-name=" + gpuName.str());

  if (debugLevel == "full") {
    options.push_back("-g");
    optimizationLevel = 0;
  } else if (debugLevel == "line-tables") {
    options.push_back("--generate-line-info");
  }

  options.push_back("--opt-level=" + std::to_string(optimizationLevel));

  return options;
}

ErrorOr<std::unique_ptr<CUDACompilationDevice>>
CUDACompilationDevice::create(StringRef targetArch) {
  return std::unique_ptr<CUDACompilationDevice>(
      new CUDACompilationDevice(targetArch.str()));
}

ErrorOr<int> CUDACompilationDevice::getApiVersion() const {
#if MODULAR_LINUX
  ErrorOr<const CUDACompilationDriver &> errOrDriver =
      CUDACompilationDriver::get();
  if (errOrDriver.isError()) {
    return errOrDriver.takeError();
  }

  // Get the NVPTX compiler version
  unsigned int majorVer, minorVer;
  auto versionResult =
      errOrDriver->nvPTXCompilerGetVersion(&majorVer, &minorVer);
  if (versionResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to get NVPTX compiler version");
  }

  // Convert NVPTX compiler version to CUDA API version format
  // CUDA version format: major * 1000 + minor * 10
  // Example: NVPTX 12.4 -> 12040
  int apiVersion = majorVer * 1000 + minorVer * 10;
  return apiVersion;
#else
  return Error("NVPTX compiler not supported on this platform");
#endif
}

ErrorOr<DriverVersion &> CUDACompilationDevice::getDriverVersion() const {
  // Initialize the driver version once on first call of this method
  // or remember the error.
  static ErrorOr<std::string> errOrVersion = []() -> ErrorOr<std::string> {
    ErrorOr<const NVML &> errOrNVML = NVML::get();
    if (errOrNVML.isError()) {
      return errOrNVML.takeError();
    }
    ErrorOr<std::string> errOrVersion = errOrNVML->GetDriverVersion();
    if (errOrVersion.isError()) {
      return errOrVersion.takeError();
    }
    return *errOrVersion;
  }();

  // Propagate the error if getting the version produced an error.
  if (errOrVersion.isError()) {
    return Error(errOrVersion.getError());
  }
  // Return the version string.
  static DriverVersion version(*errOrVersion);
  return version;
}

ErrorOr<const char *> CUDACompilationDevice::getArchName() const {
  return targetArch().c_str();
}

#if MODULAR_LINUX
// Compile PTX to CUBIN by invoking an external ptxas binary.
// This is used when MODULAR_NVPTX_COMPILER_PATH is set, allowing users on
// older NVIDIA drivers or hardware to bypass our bundled libnvptxcompiler.
//
// Adapted from:
// https://github.com/llvm/llvm-project/blob/cf5aa559a8b69/mlir/lib/Target/LLVM/NVVM/Target.cpp#L346
static ErrorOr<M::BufferRef>
compilePTXToCUBINViaPTXAS(StringRef ptxasCompilerPath, StringRef ptx,
                          StringRef gpuName,
                          ArrayRef<std::string> nvptxOptions) {
  if (!llvm::sys::fs::is_regular_file(ptxasCompilerPath))
    return Error("MODULAR_NVPTX_COMPILER_PATH points to '" + ptxasCompilerPath +
                 "' which does not exist or is not a file");

  if (!llvm::sys::fs::can_execute(ptxasCompilerPath))
    return Error("MODULAR_NVPTX_COMPILER_PATH points to '" + ptxasCompilerPath +
                 "' which is not executable");

  // Create temp files for PTX input, CUBIN output, and log output.
  llvm::SmallString<128> ptxPath, cubinPath, logPath;
  if (auto ec = llvm::sys::fs::createTemporaryFile("mojo", "ptx", ptxPath))
    return Error("Failed to create temporary PTX file: " + ec.message());
  llvm::FileRemover ptxRemover(ptxPath);

  if (auto ec = llvm::sys::fs::createTemporaryFile("mojo", "cubin", cubinPath))
    return Error("Failed to create temporary CUBIN file: " + ec.message());
  llvm::FileRemover cubinRemover(cubinPath);

  if (auto ec = llvm::sys::fs::createTemporaryFile("mojo", "log", logPath))
    return Error("Failed to create temporary log file: " + ec.message());
  llvm::FileRemover logRemover(logPath);

  // Write PTX to the temp file.
  {
    std::error_code ec;
    llvm::raw_fd_ostream ptxStream(ptxPath, ec);
    if (ec)
      return Error("Failed to write PTX to temporary file: " + ec.message());
    ptxStream << ptx;
    if (ptxStream.has_error())
      return Error("Failed to write PTX to temporary file: " +
                   ptxPath.str().str());
    ptxStream.flush();
  }

  // Build ptxas command line. Translate nvptxcompiler options to ptxas flags.
  SmallVector<StringRef, 12> ptxasArgs(
      {ptxasCompilerPath, ptxPath, "-o", cubinPath});
  for (const auto &opt : nvptxOptions)
    ptxasArgs.push_back(opt);

  // Redirect stdout and stderr to the log file.
  std::optional<StringRef> redirects[] = {
      std::nullopt,
      logPath,
      logPath,
  };

  std::string errMsg;
  int exitCode = llvm::sys::ExecuteAndWait(ptxasCompilerPath, ptxasArgs,
                                           /*Env=*/std::nullopt,
                                           /*Redirects=*/redirects,
                                           /*SecondsToWait=*/0,
                                           /*MemoryLimit=*/0,
                                           /*ErrMsg=*/&errMsg);

  if (exitCode != 0) {
    // Try to read the log for a useful error message.
    auto logBuf = llvm::MemoryBuffer::getFile(logPath);
    if (logBuf && !(*logBuf)->getBuffer().empty())
      return Error("ptxas compilation failed. Log:\n" +
                   (*logBuf)->getBuffer().str());
    if (!errMsg.empty())
      return Error("ptxas compilation failed: " + errMsg);
    return Error("ptxas compilation failed with exit code " +
                 std::to_string(exitCode));
  }

  // Read the resulting CUBIN.
  auto cubinBuf = llvm::MemoryBuffer::getFile(cubinPath);
  if (!cubinBuf)
    return Error("Failed to read compiled CUBIN from " + cubinPath.str().str() +
                 ": " + cubinBuf.getError().message());

  return M::Buffer::get((*cubinBuf)->getBuffer());
}
#endif // MODULAR_LINUX

ErrorOr<M::BufferRef>
CUDACompilationDevice::compileFunction(StringRef ptx, StringRef debugLevel,
                                       int32_t optimizationLevel,
                                       StringRef targetAccelerator) const {
#if MODULAR_LINUX
  // Use targetAccelerator if provided, otherwise use this device's target arch
  StringRef gpuName =
      targetAccelerator.empty() ? targetArch() : targetAccelerator;

  auto options = CUDACompilationDevice::setupCompilationOptionsNVPTXCompiler(
      debugLevel, optimizationLevel, gpuName);

  // If MODULAR_NVPTX_COMPILER_PATH is set, use an external ptxas binary
  // instead of the bundled libnvptxcompiler. This allows users on older NVIDIA
  // drivers or pre-Turing hardware to still compile GPU code.
  if (auto ptxasPath =
          llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_PATH")) {
    return compilePTXToCUBINViaPTXAS(*ptxasPath, ptx, gpuName, options);
  }

  // NOTE: NVPTX compiler is a standalone library that doesn't require a CUDA
  // context or GPU to be present. This allows compilation in environments
  // without GPUs (e.g., compile-only mode with virtual devices).

  ErrorOr<const CUDACompilationDriver &> errOrDriver =
      CUDACompilationDriver::get();
  if (errOrDriver.isError()) {
    return errOrDriver.takeError();
  }
  const CUDACompilationDriver &driver = *errOrDriver;

  nvPTXCompilerHandle compiler = nullptr;
  size_t elfSize, infoSize, errorSize;
  unsigned int minorVer, majorVer;

  // Get NVPTX compiler version
  auto versionResult = driver.nvPTXCompilerGetVersion(&majorVer, &minorVer);
  if (versionResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to get NVPTX compiler version");
  }

  // Create NVPTX compiler handle
  auto createResult = driver.nvPTXCompilerCreate(&compiler, (size_t)ptx.size(),
                                                 (const char *)ptx.data());
  if (createResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to create NVPTX compiler");
  }

  // Setup compilation options
  constexpr int maxNumOptions = 16;
  const char *compileOptions[maxNumOptions];
  for (auto [idx, option] : llvm::enumerate(options)) {
    compileOptions[idx] = option.c_str();
  }

  // Compile PTX to CUBIN
  CUDACompilationDriver::Result result =
      driver.nvPTXCompilerCompile(compiler, options.size(), compileOptions);

  if (result != CUDACompilationDriver::SUCCESS) {
    // Get error log
    driver.nvPTXCompilerGetErrorLogSize(compiler, &errorSize);

    std::string errorLog;
    if (errorSize != 0) {
      errorLog.resize(errorSize + 1);
      CUDACompilationDriver::Result getErrorResult =
          driver.nvPTXCompilerGetErrorLog(compiler, errorLog.data());

      if (getErrorResult != CUDACompilationDriver::SUCCESS) {
        driver.nvPTXCompilerDestroy(&compiler);
        return Error("Compilation failed and could not retrieve error log");
      }
    }

    driver.nvPTXCompilerDestroy(&compiler);
    return Error(errorLog + "\nCompilation to CUBIN failed");
  }

  // Get compiled program size
  auto sizeResult =
      driver.nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize);
  if (sizeResult != CUDACompilationDriver::SUCCESS) {
    driver.nvPTXCompilerDestroy(&compiler);
    return Error("Failed to get compiled program size");
  }

  // Allocate buffer for compiled program
  auto resultBuf = M::WriteableBuffer::get(elfSize);

  // Get compiled program
  auto programResult = driver.nvPTXCompilerGetCompiledProgram(
      compiler, (void *)resultBuf->getBufferStart());
  if (programResult != CUDACompilationDriver::SUCCESS) {
    driver.nvPTXCompilerDestroy(&compiler);
    return Error("Failed to get compiled program");
  }

  // Check for warnings
  driver.nvPTXCompilerGetInfoLogSize(compiler, &infoSize);

  std::optional<std::string> printWarning =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WARN");
  std::optional<std::string> werror =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WERROR");

  if (infoSize != 0 && (printWarning || werror)) {
    std::string infoLog;
    infoLog.resize(infoSize + 1);
    CUDACompilationDriver::Result getLogResult =
        driver.nvPTXCompilerGetInfoLog(compiler, infoLog.data());

    if (getLogResult != CUDACompilationDriver::SUCCESS) {
      driver.nvPTXCompilerDestroy(&compiler);
      return Error("Compilation has warnings but could not retrieve info log");
    }

    if (werror) {
      driver.nvPTXCompilerDestroy(&compiler);
      return Error(infoLog + "\nCompilation failed due to warnings");
    } else {
      llvm::errs() << infoLog << "\n";
    }
  }

  // Clean up compiler
  driver.nvPTXCompilerDestroy(&compiler);

  return resultBuf;
#else  // MODULAR_LINUX
  return Error("NVPTX compiler not supported on this platform");
#endif // MODULAR_LINUX
}

} // namespace M::Driver
