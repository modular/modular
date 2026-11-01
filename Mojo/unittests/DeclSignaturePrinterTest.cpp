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

#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::KGEN;

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsDropsMemberAccess) {
  EXPECT_EQ(stripImplicitArgParams(
                "UnsafePointer[Scalar[dtype], output.origin]", "output"),
            "UnsafePointer[Scalar[dtype]]");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsDropsBracketsWhenEmpty) {
  EXPECT_EQ(stripImplicitArgParams("MyStruct[arg.x]", "arg"), "MyStruct");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsPreservesPrefixOfName) {
  EXPECT_EQ(stripImplicitArgParams("Foo[outputFoo]", "output"),
            "Foo[outputFoo]");
  EXPECT_EQ(stripImplicitArgParams("Foo[output]", "output"), "Foo[output]");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsNestedBrackets) {
  EXPECT_EQ(stripImplicitArgParams("Foo[Bar[arg.x]]", "arg"), "Foo[Bar]");
  EXPECT_EQ(stripImplicitArgParams("Foo[Bar[arg.x], arg.y]", "arg"),
            "Foo[Bar]");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsMultiSegmentMemberAccess) {
  EXPECT_EQ(stripImplicitArgParams("Foo[arg.x.y]", "arg"), "Foo");
  EXPECT_EQ(stripImplicitArgParams("Foo[arg.123]", "arg"), "Foo[arg.123]");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsEmptyArgNameNoOp) {
  EXPECT_EQ(stripImplicitArgParams("Foo[arg.x]", ""), "Foo[arg.x]");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsNoBrackets) {
  EXPECT_EQ(stripImplicitArgParams("Int", "x"), "Int");
}

TEST(DeclSignaturePrinterTest, StripImplicitArgParamsCrossArgRefPreserved) {
  EXPECT_EQ(stripImplicitArgParams("Baz[a.x]", "b"), "Baz[a.x]");
}

TEST(DeclSignaturePrinterTest,
     StripImplicitArgParamsComplexExpressionPreserved) {
  EXPECT_EQ(stripImplicitArgParams("Origin[mut=output.is_mutable]", "output"),
            "Origin[mut=output.is_mutable]");
}
