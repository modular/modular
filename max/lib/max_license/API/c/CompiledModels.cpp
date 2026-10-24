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

#include "max/max_license/API/c/CompiledModels.h"
#include "GraphCompiler/Driver/Compile/CompileDriver.h"
#include "GraphCompiler/FrameworkFrontend.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace Detail;

/// Reads M::Model instances from compiledModelBuffer.
/// Returns multiple models encapsulated in a `ModelList` field in the returned
/// `UninitializedModel`.
static ErrorOr<UninitializedModel>
loadModelsFromMEF(BufferRef compiledModelBuffer) {

  auto mefFileOr = MEFFile::open(compiledModelBuffer->getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();

  RCRef<MEFFile> mefHandle = mefFileOr.takeValue();

  auto modelsOr = M::loadMGPMEFFile(mefHandle);
  if (modelsOr.isError())
    return modelsOr.takeError();

  auto models = modelsOr.takeValue();

  return UninitializedModel(std::move(compiledModelBuffer), std::move(models),
                            std::move(mefHandle));
}

/// Returns an initialized framework context.
static ErrorOr<std::unique_ptr<MOFContext>>
setupFrameworkContext(M::ContextRef context,
                      const CompilationRequest &compileConfig,
                      const M::Engine::Config &runtimeConfig) {
  std::unique_ptr<MOFContext> frameworkCtx =
      std::make_unique<MOFContext>(context);

  auto configOr = frameworkCtx->setConfigs(compileConfig.getFrameworkConfigs());
  if (configOr.isError())
    return configOr.takeError();

  return frameworkCtx;
}

/// Initializes the `ModelContext`'s framework context and source model buffer.
/// Pre-conditions:
/// - `ctx.driverApi` is initialized.
/// - `ctx.model` is initialized with a valid model path.
static ErrorOrSuccess
setModelAndFramework(M::ContextRef context, ModelContext &ctx,
                     const CompilationRequest &compileConfig,
                     const M::Engine::Config &runtimeConfig) {
  auto frameworkCtxOr =
      setupFrameworkContext(std::move(context), compileConfig, runtimeConfig);
  if (frameworkCtxOr.isError())
    return frameworkCtxOr.takeError();
  ctx.frameworkCtx = frameworkCtxOr.takeValue();

  return success();
}

ErrorOr<CompiledModels>
CompiledModels::read(BufferRef buffer, std::unique_ptr<MOFContext> frameworkCtx,
                     SourceModel sourceModel) {
  ErrorOr<UninitializedModel> uninitializedModelOr =
      loadModelsFromMEF(std::move(buffer));
  if (uninitializedModelOr.isError())
    return uninitializedModelOr.takeError();

  return CompiledModels(std::move(sourceModel), std::move(frameworkCtx),
                        uninitializedModelOr.takeValue());
}

ErrorOr<CompiledModels> CompiledModels::read(BufferRef buffer) {
  if (!isMEFFile(buffer->getBuffer()))
    return Error("buffer does not contain a MEF artifact");

  return read(std::move(buffer), /*frameworkCtx=*/nullptr,
              SourceModel(ModelSource{nullptr, FileFormat::kMEFFile}));
}

ErrorOr<CompiledModels>
CompiledModels::create(M::ContextRef context, const CompilationRequest &compCfg,
                       const M::Engine::Config &rtCfg) {
  // If model is from an opaque source with a known format.
  if (auto modelSourceOr = compCfg.getModelSource(); modelSourceOr) {
    auto modelSource = modelSourceOr.value();
    auto frameworkCtxOr = setupFrameworkContext(context.copy(), compCfg, rtCfg);
    if (frameworkCtxOr.isError())
      return frameworkCtxOr.takeError();
    std::unique_ptr<MOFContext> &frameworkCtx = *frameworkCtxOr;

    auto outputBufferOr = frameworkCtx->compileToBinary(modelSource.source,
                                                        modelSource.fileFormat);
    if (outputBufferOr.isError())
      return outputBufferOr.takeError();

    BufferRef outputBuffer = outputBufferOr.takeValue();

    return read(std::move(outputBuffer), std::move(frameworkCtx),
                SourceModel(modelSource));
  } else if (auto modelPathOr = compCfg.getModelPath(); modelPathOr) {
    // Capture runtime configuration which is also needed by the compile-time
    // configuration, since that's currently the only way to convey options
    // down into the driver's M_compileToBinary.
    auto modelPath = modelPathOr.value();

    ErrorOr<ModelContext> modelCtxOr =
        loadModelFramework(modelPath, context.copy());
    if (modelCtxOr.isError())
      return modelCtxOr.takeError();
    auto modelCtx = modelCtxOr.takeValue();
    // Convey compilation options to the driver.
    ErrorOrSuccess errorOr =
        setModelAndFramework(context.copy(), modelCtx, compCfg, rtCfg);
    if (errorOr.isError())
      return errorOr.takeError();

    SourceModel &sourceModel = modelCtx.model;
    M::BufferRef modelBuffer = modelCtx.buffer.copy();
    std::unique_ptr<MOFContext> &frameworkCtx = modelCtx.frameworkCtx;

    // If it's already compiled use that directly.
    if (modelCtx.model.format() == FileFormat::kMEFFile) {
      return read(modelBuffer.copy(), std::move(frameworkCtx), modelCtx.model);
    }

    // Actually compile the model.
    auto outputBufferOr =
        frameworkCtx->compileToBinary(modelBuffer.copy(), sourceModel.format());
    if (outputBufferOr.isError())
      return outputBufferOr.takeError();

    BufferRef outputBuffer = outputBufferOr.takeValue();
    return read(std::move(outputBuffer), std::move(frameworkCtx),
                modelCtx.model);
  }
  auto error = Error("Model path or source is not set in compilation config");
  auto *ctx = context->get<GML::MaxEngineTelemetryContext>();
  GML::emitErrorTelemetry<GML::MaxEngineErrorCode::UserInputError>(
      *ctx, error.copy());
  return error;
}
