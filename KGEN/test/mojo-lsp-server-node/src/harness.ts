import * as assert from "assert";
import * as path from "path";
import { ChildProcess, spawn } from "child_process";
import { once } from "events";
import { readFile } from "fs/promises";
import { setTimeout } from "timers/promises";
import {
  ClientCapabilities,
  CompletionItem,
  CompletionList,
  CompletionParams,
  CompletionRequest,
  DidOpenTextDocumentNotification,
  DocumentSymbol,
  DocumentSymbolRequest,
  Hover,
  HoverRequest,
  InitializeParams,
  Location,
  MessageConnection,
  Position,
  PublishDiagnosticsParams,
  Range,
  ReferencesRequest,
  SymbolInformation,
} from "vscode-languageserver-protocol";
import { createMessageConnection } from "vscode-languageserver-protocol/node";

export class LanguageServer {
  public connection: MessageConnection;
  private serverProcess: ChildProcess;
  private alive: boolean = true;

  constructor() {
    this.serverProcess = spawn(
      process.env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"]!,
      ["-wait-on-shutdown"],
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
    this.serverProcess.on("error", () => (this.alive = false));

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
    assert.ok(this.serverProcess.kill());
    const result = await Promise.race([
      once(this.serverProcess, "exit"),
      setTimeout(5000, "timeout"),
    ]);

    if (result === "timeout") {
      console.error(
        "Timed out waiting for language server to exit, did server crash?"
      );
    }
  }
}

export class Document {
  public readonly uri: string;

  public get content(): string {
    return this._content;
  }

  private _content: string;
  private _lines: string[] = [];
  private version = 0;

  constructor(private server: LanguageServer, uri: string, content: string) {
    this.uri = uri;
    this._content = content;
    this._lines = content.split("\n");
  }

  public static async fromFile(
    server: LanguageServer,
    file: string
  ): Promise<Document> {
    file = path.resolve(file);
    let content = await readFile(file, {
      encoding: "utf-8",
    });

    return new Document(server, `file://${file}`, content);
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

  /// Find the first position that a substring appears in the document. Throws
  /// if the substring is not within the document.
  public findFirstPosition(substr: string): Position {
    assert.doesNotMatch(substr, /\n/, "substr cannot contain a newline");

    for (let line = 0; line < this._lines.length; ++line) {
      let lineContent = this._lines[line];
      let offset = lineContent.indexOf(substr);
      if (offset !== -1) {
        return {
          line,
          character: offset,
        };
      }
    }

    throw new Error("substring not found in document content");
  }

  public findLastPosition(substr: string): Position {
    assert.doesNotMatch(substr, /\n/, "substr cannot contain a newline");

    for (let line = this._lines.length - 1; line > 0; --line) {
      let lineContent = this._lines[line];
      let offset = lineContent.indexOf(substr);
      if (offset !== -1) {
        return {
          line,
          character: offset,
        };
      }
    }

    throw new Error("substring not found in document content");
  }

  public findFirstRange(substr: string): Range {
    let pos = this.findFirstPosition(substr);

    return {
      start: pos,
      end: { line: pos.line, character: pos.character + substr.length },
    };
  }

  async hover(position: Position): Promise<Hover | null> {
    return this.server.connection.sendRequest(HoverRequest.type, {
      textDocument: {
        uri: this.uri,
      },
      position,
    });
  }

  public async references(
    position: Position,
    includeDeclaration: boolean
  ): Promise<Location[] | null> {
    return this.server.connection.sendRequest(ReferencesRequest.type, {
      textDocument: {
        uri: this.uri,
      },
      position,
      context: {
        includeDeclaration,
      },
    });
  }

  public async documentSymbols(): Promise<
    DocumentSymbol[] | SymbolInformation[] | null
  > {
    return this.server.connection.sendRequest(DocumentSymbolRequest.type, {
      textDocument: {
        uri: this.uri,
      },
    });
  }

  public async complete(position: Position): Promise<CompletionItem[] | null> {
    const response = await this.server.connection.sendRequest(
      CompletionRequest.type,
      {
        textDocument: {
          uri: this.uri,
        },
        position,
      }
    );

    if (Array.isArray(response) || response === null) {
      // tsc doesn't quite understand that we've narrowed the type of `response`
      // here so we have to help ita long.
      return response as CompletionItem[] | null;
    } else {
      // The protocol allows returning a CompletionList struct with a few fields
      // that we currently don't return. For convenience, we try to unwrap this
      // struct into its items array, but if/when the server starts returning
      // this information we want a signal to update the tests relying on it.
      assert.ok(
        !response.isIncomplete && response.itemDefaults === undefined,
        "CompletionList response contained unhandled fields"
      );

      return response.items;
    }
  }
}
