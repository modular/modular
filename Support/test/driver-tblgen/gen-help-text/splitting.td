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

// RUN: %driver-tblgen -gen-help-text %s | FileCheck %s

// RUN: %driver-tblgen -gen-help-text %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

// CHECK: This subcommand has a long description that contains newline characters.
// CHECK-NEXT: It also contains some extra-long words.
//
// CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
// CHECK-NEXT: tempor incididunt ut labore et dolore magna aliqua. Cursus sit amet
// CHECK-NEXT: dictum sit amet justo donec enim.
//
// CHECK: Viverra nibh cras pulvinar mattis nunc sed. Amet luctus venenatis lectus
// CHECK-NEXT: magna
// CHECK-NEXT: fringillaurnaporttitorrhoncusinmollisnuncsedidsemperrisusinhendreritacauctorauguemaurisaugue
// CHECK-NEXT: neque gravida. Sit amet risus nullam eget felis eget. Elit at imperdiet
// CHECK-NEXT: dui
// CHECK-NEXT: accumsansitametnullafacilisimorbifacilisisleovelfringillaestullamcorperegeturnaduisconvallis
// CHECK-NEXT: convallis tellus id. Diam maecenas ultricies mi eget mauris pharetra et
// CHECK-NEXT: ultrices neque. Vitae turpis massa sed elementum tempus egestas sed.
def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"splitting",
  "Test that text is split along word boundaries.",
  "This subcommand has a long description that contains newline characters. It "
  "also contains some extra-long words.\n"
  "\n"
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
  "tempor incididunt ut labore et dolore magna aliqua. Cursus sit amet dictum "
  "sit amet justo donec enim.\n"
  "\n"
  "Viverra nibh cras pulvinar mattis nunc sed. Amet luctus venenatis lectus "
  "magna "
  "fringillaurnaporttitorrhoncusinmollisnuncsedidsemperrisusinhendreritacauctorauguemaurisaugue"
  " neque gravida. Sit amet risus nullam eget felis eget. Elit at imperdiet dui "
  "accumsansitametnullafacilisimorbifacilisisleovelfringillaestullamcorperegeturnaduisconvallis"
  " convallis tellus id. Diam maecenas ultricies mi eget mauris pharetra et "
  "ultrices neque. Vitae turpis massa sed elementum tempus egestas sed.",
  [UsageInfo<"", "", /*requiresInput=*/false>]>, Index<0>;
