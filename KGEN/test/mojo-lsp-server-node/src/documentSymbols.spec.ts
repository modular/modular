import * as assert from "assert";
import { Document, LanguageServer } from "./harness";
import { SymbolKind } from "vscode-languageserver-protocol";

describe("document symbols", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should not crash when importing the current document", async function() {
    const doc = new Document(server, "test:///test.mojo", `
import .test
`);
    await doc.open();
    // Nothing more to do here; we just need the server to not crash.
  });

  it("should get document symbols", async function() {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
alias Value = 10

fn foo(a: UnsafePointer[Float32]) -> Float32:
  var variable = 15
  fn inner_fn():
    return
  fn inner_closure(arg: Int, arg2: type_of(arg)) -> Float32:
    return a.load[width=1](arg)
  return inner_fn(variable)

struct struct_name:
  fn struct_fn():
    return

  var field: Int

trait trait_name:
    fn trait_fn(self):
        ...
`
    );
    await doc.open();

    assert.partialDeepStrictEqual(await doc.documentSymbols(), [
      { name: "Value", kind: SymbolKind.Property, detail: "10" },
      {
        name: "foo",
        kind: SymbolKind.Function,
        detail: "foo(a: UnsafePointer[Float32]) -> Float32",
        children: [
          { name: "inner_fn", kind: SymbolKind.Function, detail: "inner_fn()" },
        ],
      },
      {
        name: "struct_name",
        kind: SymbolKind.Struct,
      },
      {
        name: "trait_name",
        kind: SymbolKind.Interface,
      },
    ]);
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
