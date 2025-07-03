import { ChildProcess, spawn } from "child_process";
import { ClientCapabilities, InitializeParams, MessageConnection, PublishDiagnosticsParams } from "vscode-languageserver-protocol";
import { createMessageConnection } from "vscode-languageserver-protocol/node";

export class LanguageServer {
  public connection: MessageConnection;
  private serverProcess: ChildProcess;

  constructor() {
    this.serverProcess = spawn(process.env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"]!, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.connection = createMessageConnection(this.serverProcess.stdout!, this.serverProcess.stdin!);
    this.connection.onError(error => {
      throw error;
    });

    this.connection.listen();
  }

  async initialize(capabilities?: ClientCapabilities) {
    await this.connection.sendRequest('initialize', {
      processId: process.pid,
      capabilities,
    } as InitializeParams);
  }

  async awaitDiagnostics(): Promise<PublishDiagnosticsParams> {
    return new Promise(resolve => {
      let conn = this.connection.onNotification('textDocument/publishDiagnostics', (params: PublishDiagnosticsParams) => {
        resolve(params);
        conn.dispose();
      });
    })
  }

  async awaitRequest<R>(method: string): Promise<R> {
    return new Promise(resolve => {
      let conn = this.connection.onRequest(method, (params: R) => {
        resolve(params);
        conn.dispose();
      })
    })
  }

  async stop() {
    await this.connection.sendRequest('shutdown');

    this.serverProcess.kill();
    let exitedPromise = new Promise(resolve => this.serverProcess.once("exit", resolve));
    await exitedPromise;
  }
}
