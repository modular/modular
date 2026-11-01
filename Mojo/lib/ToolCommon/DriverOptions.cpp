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

#include "Mojo/ToolCommon/DriverOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;

void M::configureSourceMgr(const State &state,
                           const DriverOptions &driverOptions,
                           llvm::SourceMgr &sourceMgr) {
  sourceMgr.setIncludeDirs(driverOptions.includePaths);
  sourceMgr.setDiagHandler(getDiagHandler(state.diagnosticFormat));
}

void M::setWarningPolicy(const State &state,
                         KGEN::FrontendOptions &frontendOptions) {
  // The warning options marshal into `state`, because State::reportWarning()
  // also reads them, and are copied from there so both places agree.
  frontendOptions.disableWarnings = state.areWarningsDisabled();
  frontendOptions.warningsAsErrors = state.areWarningsAsErrors();
}
