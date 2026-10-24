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
// This file implements the DriverPlugin interfaces within M::Driver.
//
//===----------------------------------------------------------------------===//

#include "Target/DriverPlugin.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>

// Plugin-specific extension function pointer types.
// These are not part of the SDK HAL API but are required by the plugin system
// for bridging into the MLRT C++ class hierarchy.
extern "C" {
typedef M_driver_status_t (*M_driver_device_get_native_func_ptr_t)(
    int64_t device_id, void **native_device, int64_t device_memory_limit);

typedef M_driver_status_t (*M_driver_device_arch_name_primary_func_ptr_t)(
    const char **arch_name);

typedef M_driver_status_t (*M_driver_create_compilation_device_func_ptr_t)(
    const char *arch, void **device_ptr);

typedef M_driver_status_t (*M_driver_destroy_compilation_device_func_ptr_t)(
    void *device_ptr);
}

namespace M::Driver::Plugin {

namespace {

class DriverImpl : public M::Driver::Plugin::Driver {
public:
  DriverImpl() = default;
  ~DriverImpl();
  ErrorOr<bool> isInitialized() const override { return true; }
  ErrorOr<int64_t> deviceCount() const override;

#define SETUP_FUNC(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "setup_driver_functions.inc"
#undef SETUP_FUNC

private:
  void reset_();

  int64_t id_;
  void *soHandle_;
  std::string soPath_;
  std::string apiName_;
  M_driver_driver_t driverHandle_ = nullptr;

  friend class M::Driver::Plugin::Driver;
  // Allow free functions in this namespace to access internals.
  friend ErrorOr<int> M::Driver::Plugin::loadPlugins();
  friend ErrorOr<M::Driver::DeviceRef>
  M::Driver::Plugin::getDevice(StringRef api, int id,
                               int64_t deviceMemoryLimit);
  friend int M::Driver::Plugin::deviceCount(StringRef api);
  friend bool M::Driver::Plugin::hasPluginForApi(StringRef api);
  friend ErrorOr<std::pair<std::string, std::string>>
  M::Driver::Plugin::getArchFromPlugins();
  friend ErrorOr<std::unique_ptr<CompilationDevice>>
  M::Driver::Plugin::getCompilationDevice(StringRef api, StringRef arch);
};

llvm::SmallVector<M::RCRef<DriverImpl>, 4> &driverList() {
  static llvm::SmallVector<M::RCRef<DriverImpl>, 4> drivers;
  return drivers;
}

// Map from API name to driver index in driverList().
std::unordered_map<std::string, int64_t> &apiToDriverId() {
  static std::unordered_map<std::string, int64_t> map;
  return map;
}

std::mutex &pluginMutex() {
  static std::mutex mu;
  return mu;
}

DriverImpl::~DriverImpl() {
  if (this->destroy && driverHandle_)
    this->destroy(driverHandle_);
  reset_();
}

void DriverImpl::reset_() {
#define SETUP_FUNC(name) this->name = nullptr;
#include "setup_driver_functions.inc"
#undef SETUP_FUNC
  if (soHandle_ != nullptr) {
    dlclose(soHandle_);
  }
  soHandle_ = nullptr;
  driverHandle_ = nullptr;
}

#define ASSIGN_OR_ERROR(type, var, expr)                                       \
  type var;                                                                    \
  {                                                                            \
    auto exprOr = (expr);                                                      \
    if (exprOr.isError()) {                                                    \
      return Error(std::string("Driver plugin: ") + exprOr.getError());        \
    }                                                                          \
    var = exprOr.takeValue();                                                  \
  }                                                                            \
  while (false)                                                                \
    ;

#define RETURN_IF_ERROR2(expr, message)                                        \
  do {                                                                         \
    auto exprOr = (expr);                                                      \
    if (exprOr.isError()) {                                                    \
      return Error(std::string("Driver plugin: ") + message + ": " +           \
                   exprOr.getError())                                          \
    }                                                                          \
  } while (false);

#define RETURN_IF_BAD_STATUS(activity, obj, api, ...)                          \
  do {                                                                         \
    M_driver_status_t status = obj->api(__VA_ARGS__);                          \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      char message[1024];                                                      \
      (void)obj->status_message(driverHandle_, status, message,                \
                                sizeof(message));                              \
      return Error(std::string("Failed ") + activity +                         \
                   ": Driver plugin failed " + std::string(#api) + "() on " +  \
                   obj->soPath_);                                              \
    }                                                                          \
  } while (false);

#define RETURN_IF_FAILED(condition, message)                                   \
  do {                                                                         \
    if (!(condition)) {                                                        \
      return Error(message + std::string(" for driver plugin ") + soPath_);    \
    }                                                                          \
  } while (false);

ErrorOr<int64_t> DriverImpl::deviceCount() const {
  int64_t numDevices = 0;
  RETURN_IF_BAD_STATUS("counting devices", this, device_count, driverHandle_,
                       &numDevices);
  return numDevices;
}

// Helper: find driver impl for a given API name.
DriverImpl *findDriverForApi(StringRef api) {
  auto &map = apiToDriverId();
  // TODO(DRIV-155): if the API is "gpu" just return the first plugin.
  // Currently we only use one plugin at a time, and the upper layer of the
  // stack use "gpu" as a stand-in for "device" in many places so pretending
  // plugins are "gpu" is helpful. This should be removed once the device type
  // handling is properly refactored.
  auto it = api == "gpu" ? map.begin() : map.find(api.str());
  if (it == map.end())
    return nullptr;
  auto &drivers = driverList();
  if (it->second < 0 || it->second >= int64_t(drivers.size()))
    return nullptr;
  return drivers[it->second].getPointer();
}

} // namespace

ErrorOr<int64_t> Driver::Create(std::string const &soPath) {
  // Idempotence
  for (int64_t i = 0; i < int64_t(driverList().size()); i++) {
    if (driverList()[i]->soPath_ == soPath) {
      return i;
    }
  }
  void *soHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (soHandle == nullptr) {
    return Error("Failed to open shared object: " + soPath + " (" +
                 std::string(dlerror()) + ")");
  }
  auto impl = M::RCRef<DriverImpl>::create();
  if (!impl) {
    return Error("Failed to allocate memory for driver plugin: " + soPath);
  }
  auto const driverId = int64_t(driverList().size());
  impl->id_ = driverId;
  impl->soHandle_ = soHandle;
  impl->soPath_ = soPath;

  // Load function pointers from the shared object.
#define SETUP_FUNC(fn)                                                         \
  {                                                                            \
    std::string symbol = "M_driver_" + std::string(#fn);                       \
    impl->fn = (M_driver_##fn##_func_ptr_t)dlsym(soHandle, symbol.c_str());    \
    if (impl->fn == nullptr) {                                                 \
      impl->reset_();                                                          \
      return Error("Failed to find symbol " + symbol + " in " + soPath);       \
    }                                                                          \
  }

#include "setup_driver_functions.inc"
#undef SETUP_FUNC

  // Create the driver. Tolerate M_DRIVER_STATUS_NO_DEVICES - the plugin
  // is still valid for compilation device creation even without hardware.
  {
    M_driver_version version = {.major = M_DRIVER_INTERFACE_VERSION_MAJOR,
                                .minor = M_DRIVER_INTERFACE_VERSION_MINOR,
                                .patch = M_DRIVER_INTERFACE_VERSION_PATCH};
    M_driver_status_t status = impl->create(&version, &impl->driverHandle_);
    if (status != M_DRIVER_STATUS_SUCCESS &&
        status != M_DRIVER_STATUS_NO_DEVICES) {
      // On init failure a plugin may hand back a degraded driver handle whose
      // sole purpose is to carry the error message; recover it here.
      char message[1024] = {0};
      M_driver_status_t msgStatus = impl->status_message(
          impl->driverHandle_, status, message, sizeof(message));
      std::string detail =
          msgStatus == M_DRIVER_STATUS_SUCCESS ? message : "no message";
      // Destroy the degraded handle (reset_ only clears the pointer) before
      // tearing down, so we do not leak it.
      if (impl->destroy && impl->driverHandle_)
        impl->destroy(impl->driverHandle_);
      impl->reset_();
      return Error(std::string("Failed initialization: Driver plugin failed "
                               "create() on ") +
                   soPath + ": " + detail);
    }
    if (version.major != M_DRIVER_INTERFACE_VERSION_MAJOR ||
        version.minor != M_DRIVER_INTERFACE_VERSION_MINOR) {
      impl->destroy(impl->driverHandle_);
      impl->reset_();
      return Error(std::string("Failed initialization: Driver plugin returned "
                               "incompatible version: ") +
                   soPath);
    }
  }

  // Query the plugin's API name via property("api").
  const char *apiName = nullptr;
  M_driver_status_t propStatus =
      impl->property(impl->driverHandle_, M_DRIVER_PROPERTY_API, &apiName);
  if (propStatus == M_DRIVER_STATUS_SUCCESS && apiName != nullptr) {
    impl->apiName_ = apiName;
    apiToDriverId()[impl->apiName_] = driverId;
  }

  driverList().push_back(impl);
  return driverId;
}

ErrorOr<DriverRef> Driver::Get(int64_t id) {
  if (id < 0 || id >= int64_t(driverList().size())) {
    return Error("Driver id out of range");
  }
  return DriverRef::copy(driverList()[id].getPointer());
}

// ===== Public Plugin API ===== //

MODULAR_CXX_EXPORT void ensurePluginsLoaded() {
  static auto _ = loadPlugins();
  (void)_;
}

MODULAR_CXX_EXPORT ErrorOr<int> loadPlugins() {
  std::lock_guard<std::mutex> lock(pluginMutex());

  auto envOr = llvm::sys::Process::GetEnv("MODULAR_DRIVER_PLUGINS");
  if (!envOr)
    return 0;

  std::string pluginsStr = *envOr;
  if (pluginsStr.empty())
    return 0;

  int loaded = 0;
  // Split on ':' (colon-separated list of .so paths)
  size_t start = 0;
  while (start < pluginsStr.size()) {
    size_t end = pluginsStr.find(':', start);
    if (end == std::string::npos)
      end = pluginsStr.size();
    std::string path = pluginsStr.substr(start, end - start);
    start = end + 1;

    if (path.empty())
      continue;

    auto driverIdOr = Driver::Create(path);
    if (driverIdOr.isError()) {
      // Reported, not silent: a plugin named here was asked for explicitly, so
      // its absence later surfaces as `Unsupported device api` from whichever
      // API it would have registered -- an error that names neither this path
      // nor the reason. Loading the remaining plugins still continues.
      llvm::errs() << "Failed to load driver plugin: " << driverIdOr.getError()
                   << "\n";
      continue;
    }
    loaded++;
  }
  return loaded;
}

MODULAR_CXX_EXPORT ErrorOr<M::Driver::DeviceRef>
getDevice(StringRef api, int id, int64_t deviceMemoryLimit) {
  DriverImpl *impl = findDriverForApi(api);
  if (!impl)
    return Error("No plugin registered for API: " + api.str());

  void *nativeDevice = nullptr;
  M_driver_status_t status =
      impl->device_get_native(id, &nativeDevice, deviceMemoryLimit);
  if (status != M_DRIVER_STATUS_SUCCESS || nativeDevice == nullptr)
    return Error("Plugin failed to get native device for API: " + api.str());

  // The plugin returns a Device* as void*. Copy a reference to it.
  auto *device = static_cast<M::Driver::Device *>(nativeDevice);
  return M::Driver::DeviceRef::copy(device);
}

MODULAR_CXX_EXPORT int deviceCount(StringRef api) {
  DriverImpl *impl = findDriverForApi(api);
  if (!impl)
    return 0;

  int64_t count = 0;
  M_driver_status_t status = impl->device_count(impl->driverHandle_, &count);
  if (status != M_DRIVER_STATUS_SUCCESS)
    return 0;
  return static_cast<int>(count);
}

MODULAR_CXX_EXPORT bool hasPluginForApi(StringRef api) {
  return findDriverForApi(api) != nullptr;
}

MODULAR_CXX_EXPORT ErrorOr<std::pair<std::string, std::string>>
getArchFromPlugins() {
  for (auto &driver : driverList()) {
    if (driver->apiName_.empty())
      continue;
    const char *archName = nullptr;
    M_driver_status_t status = driver->device_arch_name_primary(&archName);
    if (status == M_DRIVER_STATUS_SUCCESS && archName != nullptr) {
      return std::make_pair(driver->apiName_, std::string(archName));
    }
  }
  return Error("No plugin provided architecture information");
}

MODULAR_CXX_EXPORT ErrorOr<std::unique_ptr<CompilationDevice>>
getCompilationDevice(StringRef api, StringRef arch) {
  DriverImpl *impl = findDriverForApi(api);
  if (!impl)
    return Error("No plugin registered for API: " + api.str());

  void *devicePtr = nullptr;
  M_driver_status_t status =
      impl->create_compilation_device(arch.data(), &devicePtr);
  if (status != M_DRIVER_STATUS_SUCCESS || devicePtr == nullptr)
    return Error("Plugin failed to create compilation device for API: " +
                 api.str());

  // The plugin's destroy_compilation_device performs `delete` on the
  // CompilationDevice*. Since the plugin is loaded with RTLD_GLOBAL
  // (sharing the host process heap), the default unique_ptr deleter
  // is equivalent.
  return std::unique_ptr<CompilationDevice>(
      static_cast<CompilationDevice *>(devicePtr));
}

} // namespace M::Driver::Plugin
