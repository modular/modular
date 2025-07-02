import { InitializeParams, MessageConnection } from "vscode-languageserver";
import { createMessageConnection } from "vscode-languageserver-protocol/node";
import * as child_process from "child_process";
import { Logger } from "vscode-languageserver-protocol";

class StdErrLogger implements Logger {
  error(message: string): void {
    console.error(message);
  }
  warn(message: string): void {
    console.error(message);
  }
  info(message: string): void {
    console.error(message);
  }
  log(message: string): void {
    console.error(message);
  }
}

describe("test", () => {
  let connection: MessageConnection;
  let serverProcess: child_process.ChildProcess;

  beforeEach("start and connect to language server", () => {
    serverProcess = child_process.spawn(
      process.env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"],
      {
        stdio: ["pipe", "pipe", "inherit"],
      }
    );
    connection = createMessageConnection(
      serverProcess.stdout!,
      serverProcess.stdin!,
      new StdErrLogger()
    );
    connection.onError((e) => console.error(e));
    connection.listen();
  });

  it("initialize", async () => {
    let result = await connection.sendRequest("initialize", {
      processId: process.pid,
      capabilities: {
        window: {
          workDoneProgress: true,
        },
      },
    } as InitializeParams);
    console.log(result);
  });

  afterEach("stop language server", async () => {
    let exitedPromise = new Promise((resolve) => {
      serverProcess.once("exit", resolve);
    });

    serverProcess.kill();
    await exitedPromise;
  });
});
