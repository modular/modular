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

#include "max/max_license/API/c/Telemetry.h"

using namespace M;

static std::string getFrameworkAsString(M::FileFormat format) {
  if (format == FileFormat::kMODialect)
    return "MaxGraph";

  if (format == FileFormat::kUnknown)
    return "Unknown";

  return "Internal";
}

void M::Telemetry::emitCompiledModelType(
    Telemetry::TelemetryContext &telemetryCtx, FileFormat format) {
  std::string framework = getFrameworkAsString(format);
  auto counter = telemetryCtx.createUInt64Counter(
      "engine.load_model", M::Telemetry::Level::L1, {},
      "Model framework being loaded into the engine");

  counter.add(1, {{"framework", std::string(framework)}});
}

void M::Telemetry::emitCompileInfo(std::string_view eventName,
                                   GML::MaxEngineTelemetryContext &telemetryCtx,
                                   std::optional<Error> error) {
  if (error) {
    GML::emitErrorTelemetry<GML::MaxEngineErrorCode::CompilationError>(
        telemetryCtx, std::move(*error));
  } else {
    auto logger = telemetryCtx.getLogger("compiler");
    logger->emitL1Event(eventName, {{"max.engine.api.language",
                                     telemetryCtx.getAPILanguage()}});
  }
}

void M::Telemetry::emitPipelineName(std::string_view eventName,
                                    Telemetry::TelemetryContext &telemetryCtx,
                                    std::string_view pipelineName) {
  auto logger = telemetryCtx.getLogger("compiler");
  logger->emitL1Event(eventName, {{"max.pipeline.name", pipelineName}});
}
