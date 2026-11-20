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
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Config/Version.h"
#include "GraphCompiler/Version.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/CacheLog.h"
#include "Support/Configuration.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MAX_CONFIG(path) _X_STRINGIFY(MAX_CONFIG_SECTION) path

#ifndef MAX_CONFIG_SECTION // NOLINT(ifdef), Wundef doesn't handle the define
#error "Expected MAX_CONFIG_SECTION to be set"
#endif

std::optional<std::filesystem::path> M::getMaxCacheDir() {
  auto configOr = Config::open();
  if (configOr.isError()) {
    MODULAR_CACHE_LOG("mef") << "getMaxCacheDir: Config::open() failed\n";
    return std::nullopt;
  }

  M::Config config = configOr.takeValue();
  auto maxCacheDir = config.getValue(STRINGIFY_MAX_CONFIG(".cache_dir"));
  if (!maxCacheDir.empty()) {
    MODULAR_CACHE_LOG("mef")
        << "getMaxCacheDir: explicit cache_dir=" << maxCacheDir << "\n";
    return std::filesystem::path(maxCacheDir.str());
  }

  auto cacheDirOr = config.getModularCacheFolderPath(/*create=*/true);
  if (cacheDirOr.isError()) {
    MODULAR_CACHE_LOG("mef")
        << "getMaxCacheDir: getModularCacheFolderPath failed\n";
    return std::nullopt;
  }

  auto result = cacheDirOr.takeValue() / ".max_cache";
  MODULAR_CACHE_LOG("mef") << "getMaxCacheDir: " << result.string() << "\n";
  return result;
}

bool M::isModelIRCacheEnabled() {
  auto configOr = Config::open();

  // Disable caching if config is inaccessible.
  if (configOr.isError())
    return false;

  bool enabled =
      configOr->getValueAsBool(STRINGIFY_MAX_CONFIG(".enable_model_ir_cache"),
                               /*defaultValue=*/true);
  MODULAR_CACHE_LOG("mef") << "isModelIRCacheEnabled: "
                           << (enabled ? "true" : "false") << "\n";
  return enabled;
}

std::optional<std::filesystem::path> M::getWeightsCacheDir() {
  auto configOr = Config::open();
  if (configOr.isError())
    return std::nullopt;

  auto weightsCacheEnabled = configOr->getValueAsBool(
      /*key=*/STRINGIFY_MAX_CONFIG(".enable_weight_cache"),
      /*defaultValue=*/false);

  if (!weightsCacheEnabled)
    return std::nullopt;

  auto baseDirOr = getMaxCacheDir();
  if (baseDirOr)
    return *baseDirOr / "weights";

  // Let the API client decide.
  return std::nullopt;
}

ErrorOr<BufferRef>
M::cacheMEFTransform(StringRef frameworkName,
                     llvm::function_ref<ErrorOr<BufferRef>()> loweringFunction,
                     WriteableBufferRef key) {
  if (!isModelIRCacheEnabled()) {
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: caching disabled\n";
    return loweringFunction();
  }

  auto baseCacheDirOr = M::getMaxCacheDir();

  if (!baseCacheDirOr) {
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: no cache dir found\n";
    return loweringFunction();
  }

  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;
  MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: cache dir="
                           << baseCacheDir.string()
                           << " framework=" << frameworkName << "\n";

  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / frameworkName.str() / "mef").string(),
      M::MO::getGraphCompilerVersionString());

  if (mefBackendOr.isError()) {
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: backend creation failed\n";
    return loweringFunction();
  }

  using MEFCache = RCRef<Cache::BlobCache<Cache::TransformCacheKey>>;
  auto mefCache = MEFCache::create(mefBackendOr.takeValue());

  std::string keyHash;
  auto foundOr = mefCache->findSync(std::move(key), &keyHash);
  if (!foundOr.isError() && foundOr->has_value()) {
    MODULAR_CACHE_LOG("mef")
        << "cacheMEFTransform: CACHE HIT key=" << keyHash << "\n";
    return std::move(**foundOr);
  }

  MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: cache miss key=" << keyHash
                           << ", compiling...\n";

  auto mefOr = loweringFunction();
  if (mefOr.isError()) {
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: compilation failed\n";
    return mefOr.takeError();
  }

  BufferRef mefBuffer = mefOr.takeValue();
  MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: compilation success, size="
                           << mefBuffer->getBufferSize() << "\n";

  // Caching failures are ignored.
  auto insertErr =
      mefCache->insertKeyedSync(std::move(keyHash), mefBuffer.copy());
  if (insertErr.isError())
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: insert failed\n";
  else
    MODULAR_CACHE_LOG("mef") << "cacheMEFTransform: insert success\n";

  return mefBuffer;
}
