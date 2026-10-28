//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares how the Mojo tools resolve their shared target options
// into a compilation target.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_TARGETOPTIONS_H
#define KGEN_TOOLCOMMON_TARGETOPTIONS_H

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"

#include <string>

namespace M {

/// The target options as given on the command line, before parseTargetOptions
/// fills in defaults and checks them against each other. An empty value means
/// the option was not given.
struct TargetArgs {
  std::string triple;
  std::string cpu;
  std::string features;
  std::string abi;
  std::string march;
  std::string mcpu;
  std::string mtune;
  std::string accelerator;
  std::string mcmodel;
  std::string largeDataThreshold;
  std::string relocationModel;
};

/// Resolve `targetArgs` into the target fields of `compilationOptions`, filling
/// in the defaults for any not given. On success, `target` is populated with
/// the selected compilation target.
ErrorOrSuccess parseTargetOptions(const TargetArgs &targetArgs,
                                  KGEN::CompilationOptions &compilationOptions,
                                  MLIRContext &ctx, TargetInfoAttr &target);

} // namespace M

#endif // KGEN_TOOLCOMMON_TARGETOPTIONS_H
