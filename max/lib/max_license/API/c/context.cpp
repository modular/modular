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

#include "max/c/context.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Support/ML/DebugPrint.h"
#include "max/c/device.h"
#include "max/c/symbol_export.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/EngineContext.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"

using namespace M;

MODULAR_API_EXPORT M_RuntimeConfig *M_newRuntimeConfig() {
  auto *config = new M_RuntimeConfig();
  return config;
}

MODULAR_API_EXPORT void M_freeRuntimeConfig(M_RuntimeConfig *config) {
  delete config;
}

MODULAR_API_EXPORT void M_runtimeConfigAddDevice(M_RuntimeConfig *config,
                                                 M_Device *device) {
  config->config.addDevice({device->spec, device->context.copy()});
}

MODULAR_API_EXPORT M_RuntimeContext *
M_newRuntimeContext(const M_RuntimeConfig *config, M_Status *status) {
  M::Engine::Config runtimeConfig = config->config;

  if (runtimeConfig.getDevices().empty()) {
    // Creates CPU Device by default. `M_newDevice` sets `status` on failure,
    // and applies the same host-target-info and isCompatible() setup the
    // config's own devices get. The config copies the context ref, so the
    // handle can be released immediately.
    M_Device *cpuDevice = M_newDevice(M_HOST, /*id=*/0, status);
    if (!cpuDevice)
      return nullptr;
    runtimeConfig.addDevice({cpuDevice->spec, cpuDevice->context});
    M_freeDevice(cpuDevice);
  }

  auto runtimeCtxOr = Engine::Context::create(runtimeConfig);
  if (runtimeCtxOr.isError()) {
    M::Detail::setError(status, runtimeCtxOr.takeError());
    return nullptr;
  }
  auto *runtimeContext = new M_RuntimeContext{runtimeCtxOr.takeValue()};
  M::Detail::setOK(status);

  return runtimeContext;
}

MODULAR_API_EXPORT void M_freeRuntimeContext(M_RuntimeContext *context) {
  delete context;
}

MODULAR_API_EXPORT void M_setDebugPrintOptions(M_RuntimeContext *runtimeContext,
                                               M_ResultOutputStyle style,
                                               unsigned int precision,
                                               const char *directory) {
  auto &ctx = runtimeContext->context.maxContext;
  DebugTensorPrintOptions &options =
      ctx->emplaceIfMissing<DebugTensorPrintOptions>();

  switch (style) {
  case M_COMPACT:
    options.style = ResultOutputStyle::kCompact;
    break;
  case M_FULL:
    options.style = ResultOutputStyle::kFull;
    break;
  case M_BINARY:
    options.style = ResultOutputStyle::kBinary;
    break;
  case M_BINARY_MAX_CHECKPOINT:
    options.style = ResultOutputStyle::kBinaryMaxCheckpoint;
    break;
  default:
    options.style = ResultOutputStyle::kNone;
  }
  options.precision = precision;
  options.binaryDir = directory;
}

MODULAR_API_EXPORT void M_setMojoDefineBool(M_RuntimeContext *runtimeContext,
                                            const char *key, bool value) {
  auto &ctx = runtimeContext->context.maxContext;
  auto &options = ctx->emplaceIfMissing<KGEN::CompilationContext>();
  options.mojoDefines[key] = value;
}

MODULAR_API_EXPORT void M_setMojoDefineInt(M_RuntimeContext *runtimeContext,
                                           const char *key, int value) {
  auto &ctx = runtimeContext->context.maxContext;
  auto &options = ctx->emplaceIfMissing<KGEN::CompilationContext>();
  options.mojoDefines[key] = value;
}

MODULAR_API_EXPORT void M_setMojoDefineString(M_RuntimeContext *runtimeContext,
                                              const char *key,
                                              const char *value) {
  auto &ctx = runtimeContext->context.maxContext;
  auto &options = ctx->emplaceIfMissing<KGEN::CompilationContext>();
  options.mojoDefines[key] = value;
}
