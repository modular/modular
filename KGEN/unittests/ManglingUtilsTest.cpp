//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/ManglingUtils.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
class MangleParameterValuesTest : public ::testing::Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  MangleParameterValuesTest() { ctx.loadDialect<KGENDialect>(); }

  /// Parses a module holding one single-parameter generator named `name`, and
  /// returns that generator. A generator name is a symbol name, so `name` is
  /// spelled as MLIR string-literal contents: `\1B` for an escape character.
  GeneratorOpInterface getGenerator(StringRef name = "gen") {
    std::string source =
        ("kgen.generator @\"" + name + "\"<p: string>() { kgen.return }").str();
    modules.push_back(parseSourceString<ModuleOp>(source, &ctx));
    ModuleOp module = *modules.back();
    assert(module && "failed to parse the test generator");
    return *module.getOps<GeneratorOpInterface>().begin();
  }

  TypedAttr stringParam(StringRef value) {
    return cast<TypedAttr>(StringAttr::get(value, StringType::get(&ctx)));
  }

  /// Renders escape characters, so a failure prints legibly and an expected
  /// value needs no raw control byte written into this file.
  static std::string show(StringRef mangled) {
    std::string result;
    for (char c : mangled)
      result += (c == '\033') ? "<ESC>" : std::string(1, c);
    return result;
  }

private:
  SmallVector<OwningOpRef<ModuleOp>> modules;
};
} // namespace

TEST_F(MangleParameterValuesTest, NoParameterValuesIsTheGeneratorName) {
  EXPECT_EQ(mangleParameterValues(getGenerator(), {}), "gen");
}

// A parameter is carried as `name=value` so a symbol is unique per
// instantiation and stays readable in a linker error or a profile.
TEST_F(MangleParameterValuesTest, ParameterValuesAreNamedInTheMangling) {
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("42")})),
            "gen,p=\"42\"");
}

// `@` is encoded as an escape character followed by `A`, which is what leaves
// room for an escape character to encode as itself doubled - see
// DistinctNamesStayDistinct.
TEST_F(MangleParameterValuesTest, AtSignIsEncodedAsEscapeThenA) {
  const std::pair<StringRef, StringRef> cases[] = {
      {"@", "gen,p=\"<ESC>A\""},
      {"a@b", "gen,p=\"a<ESC>Ab\""},
      // Adjacent occurrences: an escaper stepping by a fixed stride walks past
      // the second one and leaves it in the symbol. Longer runs exercise no
      // further state here, but pin the progression against an escaper that
      // does carry some.
      {"@@", "gen,p=\"<ESC>A<ESC>A\""},
      {"a@@b", "gen,p=\"a<ESC>A<ESC>Ab\""},
      {"a@@@b", "gen,p=\"a<ESC>A<ESC>A<ESC>Ab\""},
      {"a@@@@b", "gen,p=\"a<ESC>A<ESC>A<ESC>A<ESC>Ab\""},
  };
  for (auto [value, expected] : cases)
    EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam(value)})),
              expected)
        << "for parameter value " << value.str();
}

// The invariant the encoding exists for: `@` is invalid in an ELF symbol name
// and fails the link, so none may survive however many there are. Stated
// separately from the encoding above, so it keeps holding if the encoding is
// ever changed.
TEST_F(MangleParameterValuesTest, NoAtSignSurvivesEscaping) {
  for (StringRef value :
       {"@", "@@", "a@b", "a@@b", "@a@", "@@@", "a@@@@b", "@A"}) {
    std::string mangled =
        mangleParameterValues(getGenerator(), {stringParam(value)});
    EXPECT_EQ(mangled.find('@'), std::string::npos)
        << "parameter value " << value.str() << " mangled to " << show(mangled)
        << ", which the linker rejects";
  }
}

// Encoding must stay reversible, or two instantiations collide on one symbol.
// Because `@` maps onto the escape character, an escape character already in a
// name has to be encoded too - doubled - or it is indistinguishable from an
// encoded `@`. A name arrives holding one when an already-mangled name is
// mangled again.
TEST_F(MangleParameterValuesTest, DistinctNamesStayDistinct) {
  TypedAttr param = stringParam("42");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen@"), {param})),
            "gen<ESC>A,p=\"42\"");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen\\1B"), {param})),
            "gen<ESC><ESC>,p=\"42\"");
}

// The delicate point of this encoding: `@` becomes an escape character plus
// `A`, so a literal `A` sitting right after either an `@` or an escape
// character is where a decoder could go wrong. `@A` must not encode to what an
// escape character followed by `A` encodes to, or the two decode alike.
TEST_F(MangleParameterValuesTest, LiteralAAfterAnEscapeStaysDistinct) {
  TypedAttr param = stringParam("42");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen@A"), {param})),
            "gen<ESC>AA,p=\"42\"");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen\\1BA"), {param})),
            "gen<ESC><ESC>A,p=\"42\"");
}

// A parameter value cannot deliver a raw escape character: it is rendered
// through the MLIR asm printer, which writes non-printables as hex text. So the
// doubling above is reached only through a generator's name, and a value that
// spells an escape sequence stays distinct from one holding `@` for free.
TEST_F(MangleParameterValuesTest, EscapeCharacterInAValueArrivesAsHexText) {
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("\033")})),
            "gen,p=\"\\1B\"");
}
