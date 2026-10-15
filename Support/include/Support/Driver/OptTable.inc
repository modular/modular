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
// This file uses x-macros to define an options enum and `llvm::opt::OptTable`
// based on the option definitions in a separate file, which must be located at
// `DRIVER_OPTIONS_PATH`.
//
//===------------------------------------------------------------*- C++ -*-===//

#ifndef DRIVER_OPTIONS_PATH
#error "DRIVER_OPTIONS_PATH must be defined before including this header"
#endif

namespace {
namespace options {
enum ID {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTFORVARIANTS, METAVAR,      \
               VALUES, SUBCOMMANDIDSOFFSET)                                    \
  OPT_##ID,
#include DRIVER_OPTIONS_PATH
#undef OPTION
  LastOption
};
} // namespace options
} // namespace

// `optionTables()` is static, so a TU that includes this for the enum alone
// trips -Wunused-function. Those define DRIVER_OPTIONS_ENUM_ONLY.
#ifndef DRIVER_OPTIONS_ENUM_ONLY

// The generated block references these unqualified.
using llvm::opt::DefaultVis;
using llvm::opt::HelpHidden;

// Defines `optionTables()`, returning the `llvm::opt::OptTable::Tables`
// aggregate. The generated block undefines the macro itself.
#define OPTTABLE_CODE
#include DRIVER_OPTIONS_PATH

#endif // DRIVER_OPTIONS_ENUM_ONLY

#undef DRIVER_OPTIONS_ENUM_ONLY
#undef DRIVER_INCLUDE
#undef DRIVER_OPTIONS_PATH
