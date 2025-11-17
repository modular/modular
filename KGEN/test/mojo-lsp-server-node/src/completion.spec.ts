import * as assert from "assert";
import { Document, LanguageServer } from "./harness";
import {
  CompletionItemKind,
  MarkupContent,
  Position,
} from "vscode-languageserver-protocol";

describe("completions", function () {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });

  it("should provide completions for imports", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
import p

# this is a comment
`
    );

    await doc.open();

    let completions = await doc.complete(Position.create(1, 8));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) =>
          i.label === "prelude" &&
          i.kind! === CompletionItemKind.Folder &&
          (i.documentation! as MarkupContent).value.includes(
            "Implements the prelude package"
          )
      )
    );
  });

  it("should provide completions for nested imports", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
import builtin.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(1, 15));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "bool" && i.kind! === CompletionItemKind.Module
      )
    );
  });

  it("should complete relative imports", async function () {
    let doc = await Document.fromFile(
      server,
      "KGEN/test/mojo-lsp-server-node/data/package/imports.mojo"
    );
    await doc.open();

    let completions = await doc.complete(
      doc.findFirstRange("from .aliases").end
    );
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "aliases" && i.kind! === CompletionItemKind.Module
      )
    );
  });

  it("should sort completion items", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
@fieldwise_init
struct Foo(Copyable, Movable):
  var __other: Int
  var ___another__: Int
  var __dunder__: Int
  var _sunder_: Int
  var _priv: Int
  var normal: Int
  fn foo(self): pass
  fn _foobar_(self): pass
  fn _bar(self): pass
  fn __baz__(self): pass

fn function(arg: Foo):
  arg.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(15, 6));
    assert.ok(completions);
    assert.deepStrictEqual(
      completions.map((i) => i.label),
      [
        "copy",
        "foo",
        "normal",
        "_bar",
        "_priv",
        "_foobar_",
        "_sunder_",
        "__baz__",
        "__copyinit__",
        "__del__",
        "__init__",
        "__moveinit__",
        "__dunder__",
        "__copyinit__is_trivial",
        "__del__is_trivial",
        "__moveinit__is_trivial",
        "___another__",
        "__other",
      ]
    );
  });

  it("should complete members", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
fn function(arg: Int):
    arg.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(2, 8));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "__add__" && i.kind! === CompletionItemKind.Function
      )
    );
    assert.ok(
      completions.some(
        (i) => i.label === "_mlir_value" && i.kind! == CompletionItemKind.Field
      )
    );
  });

  it("should complete at the top-level", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
fn function() -> Int:
    var value: Int = 10
    return value
`
    );
    await doc.open();

    let completions = await doc.complete(doc.findFirstPosition("nt"));
    assert.ok(completions);
    // TODO(MOTO-639): Check item.kind when we start resolving this correctly.
    assert.ok(completions.some((i) => i.label === "Int"));

    completions = await doc.complete(doc.findLastPosition("value"));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "value" && i.kind! == CompletionItemKind.Variable
      )
    );
  });

  it("should not regress MOTO-767", async function () {
    let doc = new Document(
      server,
      "test:///moto-767.mojo",
      `
fn main() raises -> :
  pass

alias T = Tuple[StringLiteral, StringLiteral, StringLiteral]

fn f[T: Equatable](s: T):
  pass
`
    );
    await doc.open();
    await doc.complete(doc.findFirstPosition("->"));
    // We need to simply not have crashed here.
  })

  describe("partial completions", function () {
    async function checkSnippet(doc: Document, startAt: string) {
      await doc.open();
      let completions = await doc.complete(doc.findFirstPosition(startAt));
      assert.ok(completions);
      assert.ok(
        completions.length > 0,
        "expected completion list to be non-empty"
      );
    }

    it("should complete for partial functions", async function () {
      let doc = new Document(
        server,
        "test:///fn_no_colon.mojo",
        `
fn function(arg: Int)`
      );
      await checkSnippet(doc, "nt");
    });

    it("should complete for partial if", async function () {
      let doc = new Document(
        server,
        "test:///if_no_colon.mojo",
        `
fn function(arg: Int):
  if arg.value`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial elif", async function () {
      let doc = new Document(
        server,
        "test:///elif_no_colon.mojo",
        `
fn function(arg: Int):
  if False:
    return
  elif arg.value`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial while", async function () {
      let doc = new Document(
        server,
        "test:///while_no_colon.mojo",
        `
fn function(arg: Int):
  while arg.value`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial with", async function () {
      let doc = new Document(
        server,
        "test:///with_no_colon.mojo",
        `
fn function(arg: Int):
  with arg.value`
      );
      await checkSnippet(doc, "value");
    });
  });
});
