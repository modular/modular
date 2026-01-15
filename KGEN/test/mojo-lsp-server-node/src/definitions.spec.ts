import assert from "assert";
import { Document, LanguageServer } from "./harness";
import { Position } from "vscode-languageserver-protocol";

describe("definitions", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should have no definitions for an empty file", async () => {
    const doc = new Document(server, "test:///test.mojo", "");
    await doc.open();
    assert.deepStrictEqual(await doc.definition(Position.create(0, 0)), []);
  });

  it("should find definitions", async () => {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
fn example():
  return 2

fn main():
  example()

  var y = 123
  print(y)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("example")),
      [
        {
          uri: doc.uri,
          range: doc.findFirstRange("example"),
        },
      ]
    );

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("y")),
      [{
        uri: doc.uri,
        range: doc.findFirstRange("y"),
      }]
    );
  });

  it("should report all definitions of overloaded functions", async () => {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
fn print(x: String):
    pass

fn print(x: Bool):
    pass

fn function[type: __TypeOfAllTypes](arg: type):
    print(arg)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("print")),
      [
        (await doc.definition(doc.findFirstPosition("print(x: String")))![0],
        (await doc.definition(doc.findFirstPosition("print(x: Bool")))![0],
      ]
    );
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
