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

#ifndef GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
#define GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H

#include "GraphCompiler/Passes.h"
#include "Support/CommandLine.h"

namespace M::MO {

/// mt command line options that influence M::MO::buildGCPipeline. See
/// GCPipelineOptions for the xxx-opt equivalents.
class GraphCompilerOptions {

public:
  std::string cacheDir{""};

  std::string defaultDeviceLabel{"cpu"};

  size_t defaultDeviceId{0};

  bool enableCompiledRegionsFusion{true};

  bool debugPrintAllTensors{false};

  bool nanCheck{false};

  size_t nanCheckStride{20};

  size_t softFusionNonHostKernelLimit{3};

  bool cacheBuffersAcrossExecutes{false};

  llvm::SmallVector<std::string> kernelLibraryPaths{};

  bool debugKGENStubs{false};

  void overrideLoweringOptions(GCPipelineOptions &gcPipelineOptions) const {

    gcPipelineOptions.cacheDir = cacheDir;
    gcPipelineOptions.defaultDeviceLabel = defaultDeviceLabel;
    gcPipelineOptions.defaultDeviceId = defaultDeviceId;
    gcPipelineOptions.enableCompiledRegionsFusion = enableCompiledRegionsFusion;
    gcPipelineOptions.debugPrintAllTensors = debugPrintAllTensors;
    gcPipelineOptions.nanCheck = nanCheck;
    gcPipelineOptions.nanCheckStride = nanCheckStride;
    gcPipelineOptions.softFusionNonHostKernelLimit =
        softFusionNonHostKernelLimit;
    gcPipelineOptions.kernelLibraryPaths = kernelLibraryPaths;
    gcPipelineOptions.debugKGENStubs = debugKGENStubs;
  }
};

class GraphCompilerCLOptions {
public:
  GraphCompilerOptions &options;
  GraphCompilerCLOptions(GraphCompilerOptions &opts) : options(opts) {}

private:
  llvm::cl::OptionCategory GraphCompilerCLOptionsCategory{
      "Graph compiler options"};

  M::cl::MOpt<std::string, true> cacheDir{
      "cache-dir",
      cl::desc("Path to CAS instance. If non-empty, large constants will be "
               "outlined into a CAS instance at the given path during "
               "compilation, and the compiled model will retrieve the "
               "constants from the same instance at model 'init' time. "
               "Default empty (ie disabled)."),
      llvm::cl::location(options.cacheDir),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<std::string, true> defaultDeviceLabel{
      "default-device-label",
      cl::desc(
          "If non-empty, assign as many ops as possible to the device "
          "with the given label (and device id based off default-device-id)."),
      llvm::cl::location(options.defaultDeviceLabel),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> defaultDeviceId{
      "default-device-id",
      cl::desc("If default-device-label is specified, this field specifies the"
               "default device id to target. Defaults to zero."),
      llvm::cl::location(options.defaultDeviceId),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableCompiledRegionsFusion{
      "enable-compiled-regions-fusion",
      cl::desc("If false, will disable compiled region fusion"),
      llvm::cl::location(options.enableCompiledRegionsFusion),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> debugPrintAllTensors{
      "debug-print-all-tensors",
      cl::desc("If enabled, print all intermediate tensors. The usual "
               "'result-output-style', 'print-precision' and "
               "'binary-output-dir' flags may be used to control the "
               "printed format and saved location."),
      llvm::cl::location(options.debugPrintAllTensors),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> nanCheck{
      "nan-check",
      cl::desc("Insert NaN/Inf checks after each compiled region's "
               "floating-point outputs."),
      llvm::cl::location(options.nanCheck),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> nanCheckStride{
      "nan-check-stride",
      cl::desc("Sample one of every N float kernel outputs when `nan-check` "
               "is enabled. Default 1 (check every tensor). Larger values "
               "trade diagnostic precision for compile time on models with "
               "hundreds of float kernel outputs."),
      llvm::cl::location(options.nanCheckStride),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> softFusionNonHostKernelLimit{
      "soft-fusion-non-host-kernel-limit",
      cl::desc(
          "Maximum number of non-host kernels to place in a single soft fused "
          "kernel."),
      llvm::cl::location(options.softFusionNonHostKernelLimit),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>>
      kernelLibraryPaths{
          "kernel-library-paths",
          cl::desc(
              "Paths to mojo packages containing mojo kernel definitions used "
              "to replace TF and MO ops."),
          llvm::cl::location(options.kernelLibraryPaths),
          llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> debugKGENStubs{
      "debug-kgen-stubs",
      cl::desc(
          "Build stubs with debuginfo so that they show up in the debugger."),
      llvm::cl::location(options.debugKGENStubs),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
