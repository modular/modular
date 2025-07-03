import { CancellationToken, DidOpenTextDocumentParams, ProgressToken, ProgressType, WorkDoneProgress, WorkDoneProgressBegin, WorkDoneProgressCreateParams, WorkDoneProgressCreateRequest, WorkDoneProgressEnd, WorkDoneProgressReport } from "vscode-languageserver-protocol";
import { LanguageServer } from "./harness"
import * as assert from "assert";

describe("progress", function() {
  let server: LanguageServer;

  beforeEach("start and connect to language server", function () {
    server = new LanguageServer();
  });

  afterEach("stop language server", async function () {
    await server.stop();
  });

  it("should respect client capabilities", async function () {
    await server.initialize({
      window: {
        workDoneProgress: undefined,
      },
    });

    server.connection.onRequest("window/workDoneProgress/create", () => {
      assert.fail("progress should not be sent");
    });

    await server.connection.sendNotification("textDocument/didOpen", {
      textDocument: {
        uri: "test:///test.mojo",
        languageId: "mojo",
        version: 0,
        text: "fn main():\n\tprint('hello world')",
      },
    } as DidOpenTextDocumentParams);

    let diagnostics = await server.awaitDiagnostics();
    assert.deepEqual([], diagnostics.diagnostics);
  });

  it("should send progress notifications when parsing", async function () {
    await server.initialize({
      window: {
        workDoneProgress: true,
      },
    });

    const tokenPromise = new Promise<ProgressToken>((resolve) => {
      let conn = server.connection.onRequest(
        WorkDoneProgressCreateRequest.method,
        (params: WorkDoneProgressCreateParams) => {
          conn.dispose();
          resolve(params.token);
          return "hi";
        }
      );
    });

    let progressFinished = tokenPromise.then((token) => {
      return new Promise<void>((resolve, reject) => {
        let stage: "initial" | "begin" | "end" = "initial";

        server.connection.onProgress(WorkDoneProgress.type, token, (params) => {
          if (params.kind === "begin") {
            if (stage !== "initial")
              return reject("expected only one `begin` progress report");

            stage = "begin";
          } else if (params.kind === "end") {
            if (stage !== "begin")
              return reject("expected `end` only after `begin` report");

            stage = "end";
            resolve();
          } else {
            return reject("should not receive `progress` reports at this time");
          }
        });
      });
    });

    await server.connection.sendNotification("textDocument/didOpen", {
      textDocument: {
        uri: "test:///test.mojo",
        languageId: "mojo",
        version: 0,
        text: "fn main():\n\tprint('hello world')",
      },
    } as DidOpenTextDocumentParams);

    await progressFinished;
  });
})
