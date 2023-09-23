//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/Host.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/Threading/HWInfo.h"
#include "Support/Threading/ThreadAffinity.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <fstream>
#include <ios>
#include <string>

#ifdef __APPLE__
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include <mach/mach_init.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#endif // __APPLE__

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif // __linux__

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

#define DEBUG_TYPE "host"

using namespace M;

#if defined(MODULAR_ARM_NEON) && defined(__APPLE__)
namespace {
// Intercept diagnostics from Clang and then bundle them up in an `Error` if
// something bad happens.
struct DiagInterceptor : public clang::DiagnosticConsumer {
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &info) override {
    if (level >= clang::DiagnosticsEngine::Level::Error) {
      // Keep the last message.
      msg.clear();
      info.FormatDiagnostic(msg);
    }
  }

  SmallString<64> msg;
};
} // namespace

static ErrorOrSuccess getCPUFeatures(const std::string &triple,
                                     const std::string &cpu,
                                     std::vector<std::string> &featureVec) {
  // Instantiate the Clang diagnostic engine. Pass in our interceptor.
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> ids(
      new clang::DiagnosticIDs());
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts(
      new clang::DiagnosticOptions());
  DiagInterceptor interceptor;
  clang::DiagnosticsEngine diags(std::move(ids), std::move(diagOpts),
                                 &interceptor, /*ShouldOwnClient=*/false);

  auto opts = std::make_shared<clang::TargetOptions>();

  opts->Triple = triple;
  opts->CPU = cpu;

  // Ask Clang to create the target info for the triple and CPU. This
  // will populate `opts` with the feature set.
  auto targetInfo = std::unique_ptr<clang::TargetInfo>(
      clang::TargetInfo::CreateTargetInfo(diags, opts));

  if (!targetInfo)
    return Error("failed to create target info: " + interceptor.msg);

  for (StringRef feature : opts->Features) {
    if (feature.front() == '+') {
      (void)feature.consume_front("+");
      featureVec.emplace_back(feature.str());
    }
  }

  return success();
}
#endif //  defined(MODULAR_ARM_NEON) && defined(__APPLE__)

//===----------------------------------------------------------------------===//
// CPU Model Info
//===----------------------------------------------------------------------===//

static ErrorOr<std::vector<std::string>> getAllHostCPUModelNames() {
#if defined(__APPLE__)
  size_t len = 0;
  if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) ==
          -1 &&
      errno != ENOMEM)
    return Error("Unable to query the machdep.cpu.brand_string for length: " +
                 llvm::Twine(strerror(errno)));
  SmallString<128> result;
  result.resize(len);
  if (sysctlbyname("machdep.cpu.brand_string", result.data(), &len, nullptr, 0))
    return Error("Unable to query the machdep.cpu.brand_string for value: " +
                 llvm::Twine(strerror(errno)));
  result.resize(len);
  return std::vector<std::string>{std::string(result)};
#elif defined(__linux__)
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (std::error_code ec = errOrBuf.getError())
    return Error("Can't open /proc/cpuinfo: " + llvm::Twine(ec.message()));
  SmallVector<StringRef> lines;
  (*errOrBuf)->getBuffer().split(lines, "\n", /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  std::vector<std::string> modelNames;
  for (StringRef line : lines) {
    auto fields = line.split(':');
    if (fields.first.trim() == "model name")
      modelNames.push_back(fields.second.trim().str());
  }
  return modelNames;
#elif defined(_MSC_VER)
  // TODO: Implement for Windows.  This is very involved -- see
  // https://github.com/Thomas-Sparber/wmi for a complete example.  (We'll need
  // to fetch Win32_Processor.Name.)  For now, return an empty vector instead
  // of returning an error.  If we return error, system-info.exe will fail even
  // if we don't care about model name.
  return std::vector<std::string>{};
#else
  return Error("Unsupported platform.");
#endif
}

ErrorOr<std::string> M::getHostCPUModelName() {
  auto allModelNamesOr = getAllHostCPUModelNames();
  if (allModelNamesOr.isError())
    return allModelNamesOr.takeError();
  auto allModelNames = std::move(*allModelNamesOr);
  std::sort(allModelNames.begin(), allModelNames.end());
  allModelNames.erase(std::unique(allModelNames.begin(), allModelNames.end()),
                      allModelNames.end());
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::interleave(allModelNames, os, ", ");
  return str;
}

//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

#ifdef __linux__
namespace {
class FileDescriptorCloser final {
public:
  explicit FileDescriptorCloser(int fd) : fd(fd) {}
  FileDescriptorCloser(const FileDescriptorCloser &) = delete;
  ~FileDescriptorCloser() {
    [[maybe_unused]] std::error_code ec =
        llvm::sys::Process::SafelyCloseFileDescriptor(fd);
    assert(!ec && "Error encountered closing read-only file descriptor, which "
                  "should never fail");
  }
  FileDescriptorCloser &operator=(const FileDescriptorCloser &) = delete;

private:
  int fd;
};
} // namespace

static M::ErrorOr<size_t> readSmallFileFromDirFD(int dirFD, const char *relPath,
                                                 llvm::Twine fileDescription,
                                                 char *buffer,
                                                 size_t bufferSize) {
  int fd = openat(dirFD, relPath, O_RDONLY);
  if (fd == -1)
    return Error("Could not open " + fileDescription + ": " + strerror(errno));
  FileDescriptorCloser fdCloser(fd);
  ssize_t nRead = read(fd, buffer, bufferSize);
  if (nRead == -1)
    return Error("Could not read " + fileDescription + ": " + strerror(errno));
  if (static_cast<size_t>(nRead) == bufferSize)
    return Error("File for " + fileDescription +
                 " too large to read into fixed-size buffer");
  return static_cast<size_t>(nRead);
}
#endif // __linux__

M::ErrorOr<size_t> M::getHostCPUCacheSize(size_t cacheLevel) {
#if defined(__APPLE__)
  size_t result;
  size_t len = sizeof(result);
  switch (cacheLevel) {
  case 1:
    if (sysctlbyname("hw.l1dcachesize", &result, &len, nullptr, 0))
      return Error("unable to query the hw.l1dcachesize");
    return result;
  case 2:
    if (sysctlbyname("hw.l2cachesize", &result, &len, nullptr, 0))
      return Error("unable to query the hw.l2cachesize");
    return result;
  default:
    return 0;
  }
#elif defined(__linux__)
  int cacheDirFD =
      open("/sys/devices/system/cpu/cpu0/cache", O_DIRECTORY | O_PATH);
  if (cacheDirFD == -1) {
    if (errno == ENOENT) {
      // There are some times, for example inside of a Docker container on Mac,
      // where the file is not there as expected. For now, hardcoding returning
      // 0 in that case, with maybe a bit of smarts in the future where the
      // host can specify what the cache is.
      return 0;
    }
    return Error("Could not open CPU0 cache directory: " +
                 llvm::Twine(strerror(errno)));
  }
  FileDescriptorCloser cacheDirFDCloser(cacheDirFD);

  for (int index = 0;; ++index) {
    char relPath[32];
    sprintf(relPath, "index%d", index);
    int cacheDirIndexFD = openat(cacheDirFD, relPath, O_DIRECTORY | O_PATH);
    if (cacheDirIndexFD == -1) {
      if (errno == ENOENT)
        break;
      return Error("Could not open cache index directory at index " +
                   llvm::Twine(index) + ": " + strerror(errno));
    }
    FileDescriptorCloser cacheDirIndexFDCloser(cacheDirIndexFD);

    char levelBuf[32], typeBuf[32], sizeBuf[32];
    auto levelLenOr =
        readSmallFileFromDirFD(cacheDirIndexFD, "level",
                               "cache index " + llvm::Twine(index) + " level",
                               levelBuf, sizeof(levelBuf));
    if (levelLenOr.isError())
      return levelLenOr.takeError();
    auto levelLen = std::move(*levelLenOr);
    auto typeLenOr = readSmallFileFromDirFD(
        cacheDirIndexFD, "type", "cache index " + llvm::Twine(index) + " type",
        typeBuf, sizeof(typeBuf));
    if (typeLenOr.isError())
      return typeLenOr.takeError();
    auto typeLen = std::move(*typeLenOr);
    auto sizeLenOr = readSmallFileFromDirFD(
        cacheDirIndexFD, "size", "cache index " + llvm::Twine(index) + " size",
        sizeBuf, sizeof(sizeBuf));
    if (sizeLenOr.isError())
      return sizeLenOr.takeError();
    auto sizeLen = std::move(*sizeLenOr);
    StringRef levelStr = StringRef(levelBuf, levelLen).trim();
    StringRef typeStr = StringRef(typeBuf, typeLen).trim();
    StringRef sizeStr = StringRef(sizeBuf, sizeLen).trim();

    size_t level;
    if (levelStr.getAsInteger(10, level))
      return Error("Could not parse cache index " + llvm::Twine(index) +
                   " level");
    if (level != cacheLevel)
      continue;

    if (typeStr != "Data" && typeStr != "Unified")
      continue;

    // Linux hard-codes the unit as K, so this should never trip unless the
    // interface is changed or something else is wrong.
    if (!sizeStr.consume_back("K"))
      return Error("Cache size at index " + llvm::Twine(index) +
                   " is not specified in K");
    size_t sizeInK;
    if (sizeStr.getAsInteger(10, sizeInK))
      return Error("Could not parse cache index " + llvm::Twine(index) +
                   " size");
    return sizeInK * 1024;
  }
  return 0;
#elif defined(_MSC_VER)

  // We can only get info for L1, L2 & L3 cache.
  if (cacheLevel >= 4)
    return 0;

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> processorInfos;
  DWORD bufferLength = 0;

  DWORD returnCode =
      GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

  if (!returnCode) {

    // This is the only error where there is a reason for retry.
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
      return Error(llvm::mapWindowsError(GetLastError()).message());
    processorInfos.resize(llvm::divideCeil(
        bufferLength, sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)));

    // Try once again with the new buffer length and pre allocated buffer.
    returnCode =
        GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

    // We can recheck for insufficient buffer length and keep on doing this
    // but it should be pretty rare to fail twice with that reason. So we will
    // bail out.
    if (!returnCode)
      return Error(llvm::mapWindowsError(GetLastError()).message());
  }

  for (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION &processorInfo :
       processorInfos) {

    if (processorInfo.Relationship == RelationCache) {
      const CACHE_DESCRIPTOR &cache = processorInfo.Cache;
      if (cache.Level == cacheLevel &&
          (cache.Type == CacheData || cache.Type == CacheUnified))
        return cache.Size;
    }
  }

  return 0;
#else
  return Error("unsupported platform");
#endif
}

//===----------------------------------------------------------------------===//
// HostMachineInfo
//===----------------------------------------------------------------------===//

static void dumpFeatures(raw_ostream &os,
                         const std::vector<std::string> &features) {
  llvm::interleaveComma(features, os);
}

static void
dumpAffinities(raw_ostream &os,
               const std::optional<std::vector<size_t>> &affinities) {
  if (affinities) {
    os << "[";
    llvm::interleave(*affinities, os, ", ");
    os << "]";
  } else {
    os << "none";
  }
}

void M::HostMachineInfo::print(llvm::raw_ostream &os) const {
  os << "target-triple: ";
  os << triple;
  os << "\nos: ";
  os << osName;
  os << "\narch: ";
  os << cpuArch;
  os << "\ncpu-model: ";
  os << cpuModelName;
  os << "\nsimd-bitwidth: ";
  os << simdBitWidth;
  os << "\nfeatures: ";
  dumpFeatures(os, cpuFeatures);
  os << "\ncore-count: ";
  os << numPhysicalCores;
  os << "\nl1-cache-size: ";
  os << l1CacheSize;
  os << "\nl2-cache-size: ";
  os << l2CacheSize;
  os << "\nl3-cache-size: ";
  os << l3CacheSize;
  os << "\nl4-cache-size: ";
  os << l4CacheSize;
  os << "\naffinities: ";
  dumpAffinities(os, affinities);
  os << "\n";
}

void M::HostMachineInfo::print(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("target-triple", triple);
  json.attribute("os", osName);
  json.attribute("arch", cpuArch);
  json.attribute("cpu-model", cpuModelName);
  json.attribute("simd-bitwidth", simdBitWidth);
  json.attribute("features", cpuFeatures);
  json.attribute("core-count", numPhysicalCores);
  json.attribute("l1-cache-size", l1CacheSize);
  json.attribute("l2-cache-size", l2CacheSize);
  json.attribute("l3-cache-size", l3CacheSize);
  json.attribute("l4-cache-size", l4CacheSize);
  if (affinities) {
    json.attribute("affinities", *affinities);
  }
  json.objectEnd();
}

void HostMachineInfo::print(HostProperty property,
                            llvm::json::OStream &json) const {
  switch (property) {
  case HostProperty::TargetTriple:
    json.attribute("target-triple", triple);
    break;
  case HostProperty::OS:
    json.attribute("os", osName);
    break;
  case HostProperty::Arch:
    json.attribute("arch", cpuArch);
    break;
  case HostProperty::CPUModel:
    json.attribute("cpu-model", cpuModelName);
    break;
  case HostProperty::SIMDBitWidth:
    json.attribute("simd-bitwidth", simdBitWidth);
    break;
  case HostProperty::Features:
    json.attribute("features", cpuFeatures);
    break;
  case HostProperty::CoreCount:
    json.attribute("core-count", numPhysicalCores);
    break;
  case HostProperty::L1CacheSize:
    json.attribute("l1-cache-size", l1CacheSize);
    break;
  case HostProperty::L2CacheSize:
    json.attribute("l2-cache-size", l2CacheSize);
    break;
  case HostProperty::L3CacheSize:
    json.attribute("l3-cache-size", l3CacheSize);
    break;
  case HostProperty::L4CacheSize:
    json.attribute("l4-cache-size", l4CacheSize);
    break;
  case HostProperty::Affinities:
    if (affinities)
      json.attribute("affinities", *affinities);
    break;
  }
}

void HostMachineInfo::print(HostProperty property,
                            llvm::raw_ostream &os) const {
  switch (property) {
  case HostProperty::TargetTriple:
    os << triple;
    break;
  case HostProperty::OS:
    os << osName;
    break;
  case HostProperty::Arch:
    os << cpuArch;
    break;
  case HostProperty::CPUModel:
    os << cpuModelName;
    break;
  case HostProperty::SIMDBitWidth:
    os << simdBitWidth;
    break;
  case HostProperty::Features:
    dumpFeatures(os, cpuFeatures);
    break;
  case HostProperty::CoreCount:
    os << numPhysicalCores;
    break;
  case HostProperty::L1CacheSize:
    os << l1CacheSize;
    break;
  case HostProperty::L2CacheSize:
    os << l2CacheSize;
    break;
  case HostProperty::L3CacheSize:
    os << l3CacheSize;
    break;
  case HostProperty::L4CacheSize:
    os << l4CacheSize;
    break;
  case HostProperty::Affinities:
    dumpAffinities(os, affinities);
    break;
  }
  os << "\n";
}

void HostMachineInfo::printStaticInfo(raw_ostream &os) const {
  print(HostProperty::TargetTriple, os);
  print(HostProperty::OS, os);
  print(HostProperty::Arch, os);
  print(HostProperty::CPUModel, os);
  print(HostProperty::SIMDBitWidth, os);
  print(HostProperty::Features, os);
  print(HostProperty::L1CacheSize, os);
  print(HostProperty::L2CacheSize, os);
  print(HostProperty::L3CacheSize, os);
  print(HostProperty::L4CacheSize, os);
}

HostMachineInfo HostMachineInfo::fromTargetInfo(const TargetInfo &targetInfo) {
  HostMachineInfo result;
  result.triple = targetInfo.triple.str();
  result.osName = llvm::Triple::getOSTypeName(targetInfo.triple.getOS());
  result.cpuArch = targetInfo.cpu;
  result.cpuFeatures = targetInfo.features;
  return result;
}

static M::ErrorOr<HostMachineInfo> getHostMachineInfoImpl() {
  HostMachineInfo machineInfo;

  machineInfo.triple = llvm::sys::getDefaultTargetTriple();
  machineInfo.osName =
      llvm::Triple::getOSTypeName(llvm::Triple(machineInfo.triple).getOS());
  machineInfo.cpuArch = llvm::sys::getHostCPUName();
  auto cpuModelNameOr = getHostCPUModelName();
  if (cpuModelNameOr.isError())
    return cpuModelNameOr.takeError();
  machineInfo.cpuModelName = std::move(*cpuModelNameOr);

  // TODO: Reconcile with getHostCPUFeatures() in MArchTarget.
  llvm::StringMap<bool> features;
  auto gotfeatures = llvm::sys::getHostCPUFeatures(features);
  if (!gotfeatures) {
    // getCPUFeatures doesn't do anything for M1. So let's ask clang.
#if defined(MODULAR_ARM_NEON) && defined(__APPLE__)
    if (auto err = getCPUFeatures(machineInfo.triple, machineInfo.cpuArch,
                                  machineInfo.cpuFeatures))
      return err.takeError();
#else
    return Error("Failed to get cpu features");
#endif //  defined(MODULAR_ARM_NEON) && defined(__APPLE__)
  }

  for (const auto &feature : features)
    if (feature.getValue())
      machineInfo.cpuFeatures.push_back(feature.getKey().str());
  llvm::sort(machineInfo.cpuFeatures);

  machineInfo.simdBitWidth = simdWidthFromFeatures(machineInfo.cpuFeatures);

  auto physicalCoresOr = M::getNumPhysicalCores();
  if (physicalCoresOr.isError())
    return physicalCoresOr.takeError();
  machineInfo.numPhysicalCores = physicalCoresOr.takeValue();

  auto l1CacheSizeOr = getHostCPUCacheSize(1);
  if (l1CacheSizeOr.isError())
    return l1CacheSizeOr.takeError();
  machineInfo.l1CacheSize = std::move(*l1CacheSizeOr);

  auto l2CacheSizeOr = getHostCPUCacheSize(2);
  if (l2CacheSizeOr.isError())
    return l2CacheSizeOr.takeError();
  machineInfo.l2CacheSize = std::move(*l2CacheSizeOr);

  auto l3CacheSizeOr = getHostCPUCacheSize(3);
  if (l3CacheSizeOr.isError())
    return l3CacheSizeOr.takeError();
  machineInfo.l3CacheSize = std::move(*l3CacheSizeOr);

  auto l4CacheSizeOr = getHostCPUCacheSize(4);
  if (l4CacheSizeOr.isError())
    return l4CacheSizeOr.takeError();
  machineInfo.l4CacheSize = std::move(*l4CacheSizeOr);

  if (haveThreadAffinity()) {
    ErrorOr<CPUSystemInfo> errOrSysInfo = CPUSystemInfo::get();
    if (!errOrSysInfo.isError()) {
      machineInfo.affinities =
          errOrSysInfo->getPreferredCpuIDs(machineInfo.numPhysicalCores);
    }
    // else: ignore error, leave field empty to denote affinities are not avail.
  }

  return std::move(machineInfo);
}

M::ErrorOr<HostMachineInfo> M::getHostMachineInfo() {
  // We cache the host machine information to make query a lot faster, since it
  // will not change between invocations.
  static M::ErrorOr<HostMachineInfo> hostMachineInfo = getHostMachineInfoImpl();
  if (hostMachineInfo.isError())
    return M::Error(hostMachineInfo.getError());
  return *hostMachineInfo;
}
