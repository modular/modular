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

#ifndef GENERICML_API_TELEMETRY_H
#define GENERICML_API_TELEMETRY_H

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "Support/SymbolExport.h"

#include <string_view>

namespace M::Telemetry {

// Logs the type of model being compiled.
void emitCompiledModelType(Telemetry::TelemetryContext &telemetryCtx,
                           FileFormat format);

MODULAR_CXX_EXPORT void
emitCompileInfo(std::string_view eventName,
                GML::MaxEngineTelemetryContext &telemetryCtx,
                std::optional<Error> error);

// Logs the pipeline name if call originates with MAX Graph.
void emitPipelineName(std::string_view eventName,
                      Telemetry::TelemetryContext &telemetryCtx,
                      std::string_view pipelineName);

} // namespace M::Telemetry

#endif // GENERICML_API_TELEMETRY_H
