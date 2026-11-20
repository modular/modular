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

// RUN: %driver-tblgen -gen-help-texts %s | FileCheck %s

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that the help texts are output as one function.",
  "This command has a visible option and a hidden option.",
  [UsageInfo<"options", "", /*variadicInput=*/false>]>;

def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

def shown : Flag<["--"], "shown">, Group<FirstOptionGroup>,
  HelpText<"Appears in both help texts.">;

def hidden : Flag<["--"], "hidden">, Group<FirstOptionGroup>,
  Flags<[HelpHidden]>, HelpText<"Appears in the hidden help text only.">;

// CHECK:      namespace {
// CHECK-NEXT: namespace options {
// CHECK:      inline M::HelpTexts helpTexts() {
// CHECK-NEXT:   return M::HelpTexts(
// CHECK-NEXT: u8R"(NAME
// CHECK:      --shown
// CHECK-NOT:  --hidden
// CHECK:      )",
// CHECK-NEXT: u8R"(NAME
// CHECK:      --shown
// CHECK:      --hidden
// CHECK:      )");
// CHECK-NEXT: }
// CHECK-NEXT: } // namespace options
// CHECK-NEXT: } // namespace
