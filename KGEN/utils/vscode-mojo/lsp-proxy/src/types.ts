//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import type {
  CodeActionParams,
  CompletionParams,
  DefinitionParams,
  DocumentSymbolParams,
  FoldingRangeParams,
  HoverParams,
  InitializeParams,
  InlayHintParams,
  ReferenceParams,
  SemanticTokensDeltaParams,
  SemanticTokensParams,
  SignatureHelpParams,
} from 'vscode-languageserver-protocol';
import type { createConnection as createClientConnection } from 'vscode-languageserver/node';

/**
 * A generic disposable.
 *
 * Note: We can't use vscode.Disposable because the proxy can't depend on the
 * VSCode API.
 */
export interface Disposable {
  dispose(): void;
}
/**
 * Type alias for a URI.
 */
export type URI = string;

/**
 * The type that represents a connection with the VSCode LSP client.
 */
export type Client = ReturnType<typeof createClientConnection>;

/**
 * This type represents the initialization options send by the extension to the
 * proxy.
 */
export interface InitializationOptions {
  /**
   * The path to `mojo-lsp-server`.
   */
  serverPath: string;
  /**
   * The arguments to use when invoking `mojo-lsp-server`.
   */
  serverArgs: string[];
  /**
   * The environment to use when invoking `mojo-lsp-server`.
   */
  serverEnv: { [env: string]: Optional<string> };
}

/**
 * This type represents a decoded JSON object.
 */
export type JSONObject = {
  [key: string]: any;
};

/**
 * A simple type alias for a LSP request id.
 */
export type RequestId = number;

/**
 * This type contains a process exit information. At least one of these two
 * fields is guaranteed not to be null.
 */
export type ExitStatus = {
  code: number | null;
  signal: NodeJS.Signals | null;
};

// The `shutdown` request doesn't have params, but using `{}` simplifies
// typechecking.
export type ShutdownParams = {};

/**
 * A custom request sent by the extension to record all traffic sent to
 * the underlying language server in a replayable form.
 */
export type RecordSessionParams = {
  /*
   * Whether to enable or disable session recording.
   */
  enabled: boolean;
};

/**
 * This union type represents all supported request params that contain a
 * `textDocument` entry.
 */
export type RequestParamsWithDocument =
  | CodeActionParams
  | CompletionParams
  | DefinitionParams
  | DocumentSymbolParams
  | FoldingRangeParams
  | HoverParams
  | ReferenceParams
  | SignatureHelpParams
  | InlayHintParams
  | SemanticTokensParams
  | SemanticTokensDeltaParams;

/**
 * This union type represents all supported request params that don't contain a
 * `textDocument` entry.
 */
export type RequestParamsWithoutDocument = InitializeParams | ShutdownParams;

/**
 * This union type represents all supported request params.
 */
export type RequestParams =
  | RequestParamsWithDocument
  | RequestParamsWithoutDocument;

export type Optional<T> = T | undefined;
