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

#include "max/max_license/API/c/detail/ModelContext.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Support/FileUtilities.h"

using namespace M;
using namespace M::Detail;

// Constructs a `ModelContext` given a model path
static ErrorOr<ModelContext>
constructModelContext(std::filesystem::path modelPath, M::ContextRef context) {
  if (std::error_code ec; !std::filesystem::exists(modelPath, ec))
    return Error("model at " + modelPath.string() + " does not exist");
  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> file = mlir::openInputFile(
      modelPath.string(), maxAttributeAlignment(), &errorMsg);

  if (!file)
    return Error(errorMsg);
  auto format =
      sniffInputFormat(modelPath.extension().string(), file->getBuffer());

  BufferRef modelBuffer = Buffer::take(std::move(file));

  auto frameworkCtx = std::make_unique<MOFContext>(std::move(context));

  return ModelContext(std::move(modelBuffer), std::move(modelPath), format,
                      std::move(frameworkCtx));
}

/// Allocates a `ModelContext` and initializes it for the framework
/// corresponding to the model given by `modelPath`.
ErrorOr<ModelContext>
M::Detail::loadModelFramework(const std::filesystem::path &modelPath,
                              M::ContextRef context) {
  auto *telemetryCtx = context->get<GML::MaxEngineTelemetryContext>();
  std::error_code ec;

  if (!std::filesystem::is_directory(modelPath, ec) && ec) {
    /// In the error message we are sending over telemetry we skip the
    /// path in user's filesystem, but include it in the one we return
    /// back to the user.
    auto error = Error("invalid path");
    GML::emitErrorTelemetry<GML::MaxEngineErrorCode::FileNotFoundError>(
        *telemetryCtx, std::move(error));
    return Error(Twine("invalid path: ") + modelPath.string());
  }
  return constructModelContext(modelPath, std::move(context));
}
