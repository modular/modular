import * as assert from "assert";
import { ChildProcess, spawn } from "child_process";
import {
  ClientCapabilities,
  DidOpenTextDocumentNotification,
  InitializeParams,
  MessageConnection,
  PublishDiagnosticsParams,
} from "vscode-languageserver-protocol";
import { createMessageConnection } from "vscode-languageserver-protocol/node";

export class LanguageServer {
  public connection: MessageConnection;
  private serverProcess: ChildProcess;
  private alive: boolean = true;

  constructor() {
    this.serverProcess = spawn(
      process.env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"]!,
      {
        stdio: ["pipe", "pipe", "inherit"],
      }
    );

    this.connection = createMessageConnection(
      this.serverProcess.stdout!,
      this.serverProcess.stdin!
    );
    this.connection.onError((err) => {
      console.error(err);
      this.alive = false;
    });

    this.connection.onClose(() => (this.alive = false));
    this.connection.onDispose(() => (this.alive = false));
    this.serverProcess.on("exit", () => (this.alive = false));

    this.connection.listen();
  }

  async initialize(capabilities?: ClientCapabilities) {
    await this.connection.sendRequest("initialize", {
      processId: process.pid,
      capabilities,
    } as InitializeParams);
  }

  async awaitDiagnostics(): Promise<PublishDiagnosticsParams> {
    return new Promise((resolve) => {
      let conn = this.connection.onNotification(
        "textDocument/publishDiagnostics",
        (params: PublishDiagnosticsParams) => {
          resolve(params);
          conn.dispose();
        }
      );
    });
  }

  async awaitRequest<R>(method: string): Promise<R> {
    return new Promise((resolve) => {
      let conn = this.connection.onRequest(method, (params: R) => {
        resolve(params);
        conn.dispose();
      });
    });
  }

  async stop() {
    assert.ok(this.alive, "server terminated early");

    await this.connection.sendRequest("shutdown");

    this.serverProcess.kill();
    let exitedPromise = new Promise((resolve) =>
      this.serverProcess.once("exit", resolve)
    );
    await exitedPromise;
  }
}

export class Document {
  public readonly uri: string;

  public get content(): string {
    return this._content;
  }

  private _content: string;
  private version = 0;

  constructor(private server: LanguageServer, uri: string, content: string) {
    this.uri = uri;
    this._content = content;
  }

  async open() {
    return await this.server.connection.sendNotification(
      DidOpenTextDocumentNotification.type,
      {
        textDocument: {
          uri: this.uri,
          languageId: "mojo",
          version: this.version,
          text: this._content,
        },
      }
    );
  }
}
