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

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace LIT;
using namespace mlir;
using namespace testing;

namespace {
class ParamTypePrintDedupTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  ParamTypePrintDedupTest() {
    ctx.loadDialect<KGENDialect>();
    ctx.loadDialect<POP::POPDialect>();
    ctx.loadDialect<LIT::LITDialect>();
  }

  /// Builds a_i = add(a_{i-1}, a_{i-1}), a shared DAG whose tree expansion is
  /// 2^depth and whose deduplicated printing is linear in `depth`.
  TypedAttr buildSharedAddDag(int depth) {
    auto indexType = IndexType::get(&ctx);
    TypedAttr level =
        ParamIndexRefAttr::get(/*depth=*/0, /*index=*/0, indexType);
    for (int i = 0; i < depth; ++i) {
      SmallVector<TypedAttr> operands = {level, level};
      level = ParamOperatorAttr::getFromBytecode(POC::Add, operands, indexType);
    }
    return level;
  }
};
} // namespace

// `!pop.array` keeps the active printer for its size and element-type
// parameters. Before that printer was threaded through, the type printed
// through a fresh diagnostic printer, so a shared DAG behind the element type
// re-expanded here even though the enclosing expression was deduplicated.
TEST_F(ParamTypePrintDedupTest, ArrayElementTypeKeepsDedupLabels) {
  constexpr int depth = 12;
  auto arrayType =
      POP::ArrayType::get(8, ParamType::get(buildSharedAddDag(depth)));

  std::string printed = getTypeAsString(arrayType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  // A re-expanded DAG would be ~2^depth leaves wide; the deduplicated form is
  // linear in `depth`.
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the element type was re-expanded: "
      << printed.size() << " chars";
}

// Same check for a resolved `!pop.union`, whose member types are a parameter
// list rather than a single type.
TEST_F(ParamTypePrintDedupTest, UnionMemberTypesKeepDedupLabels) {
  constexpr int depth = 12;
  auto unionType =
      POP::UnionType::get({ParamType::get(buildSharedAddDag(depth))});

  std::string printed = getTypeAsString(unionType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; a member type was re-expanded: "
      << printed.size() << " chars";
}

// The array's own size parameter is a parameter value, so it must also print
// through the active printer rather than a fresh one.
TEST_F(ParamTypePrintDedupTest, ArraySizeKeepsDedupLabels) {
  auto indexType = IndexType::get(&ctx);
  auto arrayType = POP::ArrayType::get(buildSharedAddDag(12), Type(indexType));

  std::string printed = getTypeAsString(arrayType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the size was re-expanded: "
      << printed.size() << " chars";
}

// `!lit.meta` wraps a single type, which can be a parameter expression. Its
// parser takes a full type, so the `!kgen.param<...>` wrapper has to survive
// printing: dropping it (as `printParamType` does) yields text that no longer
// parses.
TEST_F(ParamTypePrintDedupTest, MetaTypeKeepsDedupLabels) {
  auto meta = LIT::MetaType::get(ParamType::get(buildSharedAddDag(12)));

  std::string printed = getTypeAsString(meta).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_NE(std::string::npos, printed.find("!kgen.param<"))
      << "the ParamType wrapper was dropped, so this cannot re-parse: "
      << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the metatype was re-expanded: "
      << printed.size() << " chars";
}

// A KGEN type wrapped in `!lit.meta` has to print through the active printer,
// which emits only the body, so the dialect prefix must be added back exactly
// once.
TEST_F(ParamTypePrintDedupTest, MetaTypeOfStructKeepsDedupLabels) {
  auto elementType = ParamType::get(buildSharedAddDag(12));
  auto structType = KGEN::StructType::get(ArrayRef<Type>{elementType},
                                          /*isMemoryOnly=*/false);
  auto meta = LIT::MetaType::get(structType);

  std::string printed = getTypeAsString(meta).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_NE(std::string::npos, printed.find("!kgen.struct<"))
      << "the dialect prefix was dropped: " << printed;
  EXPECT_EQ(std::string::npos, printed.find("!kgen.!kgen."))
      << "the dialect prefix was emitted twice: " << printed;
  EXPECT_LT(printed.size(), 1200u)
      << "printed type is too large; the nested type was re-expanded: "
      << printed.size() << " chars";
}

// `!lit.trait` holds trait symbol references, each with its own parameter
// values, plus an optional per-symbol constraint whose proposition is itself a
// parameter expression. Both must keep the active printer.
TEST_F(ParamTypePrintDedupTest, TraitSymbolParamValuesKeepDedupLabels) {
  auto symbol = SymbolRefAttr::get(&ctx, "MyTrait");
  auto traitSymbol = TraitSymbolAttr::get(
      &ctx, symbol, ArrayRef<TypedAttr>{buildSharedAddDag(12)});
  auto traitType = LIT::TraitType::get(
      &ctx, ArrayRef<TraitSymbolAttr>{traitSymbol}, ArrayRef<ConstraintAttr>{});

  std::string printed = getTypeAsString(traitType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the trait parameter values were "
         "re-expanded: "
      << printed.size() << " chars";
}

// `!lit.origin<isMutable>` is a parameter value, so a parametric origin must
// keep the active printer too.
TEST_F(ParamTypePrintDedupTest, OriginKeepsDedupLabels) {
  auto boolType = SIMDType::getScalarBoolType(&ctx);
  TypedAttr level = ParamIndexRefAttr::get(/*depth=*/0, /*index=*/0, boolType);
  for (int i = 0; i < 12; ++i) {
    SmallVector<TypedAttr> operands = {level, level};
    level = ParamOperatorAttr::getFromBytecode(POC::And, operands, boolType);
  }
  auto originType = LIT::OriginType::get(level);

  std::string printed = getTypeAsString(originType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; isMutable was re-expanded: "
      << printed.size() << " chars";
}

// The types below are registered through `registerMnemonicType`, which derives
// their param-aware printer from the type's own
// `print(AsmPrinter &, KGENParamPrinter &)`. Each must still route its nested
// parameters through the active printer, and must emit its mnemonic exactly
// once.

// `!kgen.pointer<elementType>`: the element type is a parameter type.
TEST_F(ParamTypePrintDedupTest, PointerElementTypeKeepsDedupLabels) {
  auto pointerType = PointerType::get(ParamType::get(buildSharedAddDag(12)));

  std::string printed = getTypeAsString(pointerType).str();

  EXPECT_EQ(std::string::npos, printed.find("pointerpointer"))
      << "the mnemonic was emitted twice: " << printed;
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the element type was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.param_list<elementType>`: same shape as a pointer, one element type.
TEST_F(ParamTypePrintDedupTest, ParamListElementTypeKeepsDedupLabels) {
  auto listType = ParamListType::get(ParamType::get(buildSharedAddDag(12)));

  std::string printed = getTypeAsString(listType).str();

  EXPECT_EQ(std::string::npos, printed.find("param_listparam_list"))
      << "the mnemonic was emitted twice: " << printed;
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the element type was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.variant<...>`: the member types are a parameter list.
TEST_F(ParamTypePrintDedupTest, VariantMemberTypesKeepDedupLabels) {
  auto variantType =
      VariantType::get(ArrayRef<Type>{ParamType::get(buildSharedAddDag(12))});

  std::string printed = getTypeAsString(variantType).str();

  EXPECT_EQ(std::string::npos, printed.find("variantvariant"))
      << "the mnemonic was emitted twice: " << printed;
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; a member type was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.typevalue<...>` holds one parameter value.
TEST_F(ParamTypePrintDedupTest, TypeValueKeepsDedupLabels) {
  auto typeValue = TypeValueType::get(buildSharedAddDag(12));

  std::string printed = getTypeAsString(typeValue).str();

  EXPECT_EQ(std::string::npos, printed.find("typevaluetypevalue"))
      << "the mnemonic was emitted twice: " << printed;
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the type value was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.simd<size, dtype>` prints through `printSIMDBody` from the registry
// rather than through a type interface, so a parametric size must still keep
// the active printer.
TEST_F(ParamTypePrintDedupTest, SIMDSizeKeepsDedupLabels) {
  auto simdType = SIMDType::get(buildSharedAddDag(12), KGENDType::kBool);

  std::string printed = getTypeAsString(simdType).str();

  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 800u)
      << "printed type is too large; the size was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.struct<...>` nests its element types in a parameter list, so it must
// keep the active printer for them as well.
TEST_F(ParamTypePrintDedupTest, StructElementTypesKeepDedupLabels) {
  auto elementType = ParamType::get(buildSharedAddDag(12));
  auto structType = KGEN::StructType::get(ArrayRef<Type>{elementType},
                                          /*isMemoryOnly=*/false);

  std::string printed = getTypeAsString(structType).str();

  EXPECT_EQ(std::string::npos, printed.find("structstruct"))
      << "the mnemonic was emitted twice: " << printed;
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  EXPECT_LT(printed.size(), 1200u)
      << "printed type is too large; an element type was re-expanded: "
      << printed.size() << " chars";
}

// `!kgen.func.literal<...>` is not registered as a pretty type: it prints with
// its dialect prefix from a special case in `printKGENType`, so it needs its
// own coverage that the prefix is emitted exactly once.
//
// The nested parameter still re-expands, because `printFuncType` prints its
// argument types with a bare `AsmPrinter` rather than the active printer. That
// gap is older than this dispatch and lives in the `FuncType` special case, not
// here.
// TODO(MOCO-5257): thread the printer through `printFuncType` so a function
// literal's signature keeps dedup labels too.
TEST_F(ParamTypePrintDedupTest, FuncLiteralKeepsDialectPrefix) {
  auto funcType =
      FuncType::get(&ctx, {ParamType::get(buildSharedAddDag(12))}, {});
  auto symbol =
      FuncSymbolAttr::get(SymbolRefAttr::get(&ctx, "lit_fn"), funcType,
                          /*paramValues=*/{});
  auto literal = FuncLiteralType::get(symbol);

  std::string printed = getTypeAsString(literal).str();

  EXPECT_EQ(std::string::npos, printed.find("!kgen.!kgen."))
      << "the dialect prefix was emitted twice: " << printed.substr(0, 120);
  EXPECT_EQ(0u, printed.rfind("!kgen.func.literal<", 0))
      << "expected exactly one dialect prefix, got: " << printed.substr(0, 120);
}

// A param-aware printer has to produce text its own parser accepts. `!lit.meta`
// takes a full type, so printing it through `printParamType` (which drops the
// `!kgen.param<...>` wrapper) silently produced IR that could not be read back.
// The expression is deliberately unshared: `getTypeAsString` uses the mangling
// printer, whose dedup labels are not parseable text.
TEST_F(ParamTypePrintDedupTest, MetaTypeRoundTripsThroughParser) {
  auto meta = LIT::MetaType::get(ParamType::get(buildSharedAddDag(1)));
  std::string printed = "!lit." + getTypeAsString(meta).str();

  std::string source =
      "kgen.generator @f() {\n  kgen.param.declare T: " + printed +
      " = <*\"type\">\n  hlcf.unreachable\n}\n";
  auto module = parseSourceString<ModuleOp>(source, &ctx);

  ASSERT_TRUE(module) << "printed type does not re-parse: " << printed;
}
