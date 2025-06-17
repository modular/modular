//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { ChildProcess, spawn } from 'child_process';
import { firstValueFrom, Subject } from 'rxjs';

import { DisposableCallback, DisposableContext } from './DisposableContext';
import {
  JSONRPCStream,
  LineSeparatedStream,
  ProcessExitStream,
} from './streams';
import {
  ExitStatus,
  InitializationOptions,
  JSONObject,
  Optional,
  RequestId,
  RequestParams,
} from './types';

const protocolHeader = 'Content-Length: ';
const protocolLineSeparator = '\r\n\r\n';

type PendingRequest = {
  params: RequestParams;
  responseStream: Subject<JSONObject>;
};

/**
 * This class manages an instance of the mojo-lsp-server process, as well as
 * supporting utilities for sending requests and notifications.
 */
export class MojoLSPServer extends DisposableContext {
  private serverProcess: ChildProcess;
  private lastSentRequestId: RequestId = -1;
  private pendingRequests = new Map<RequestId, PendingRequest>();
  /**
   * @param initializationOptions The options needed to spawn the
   *     mojo-lsp-server.
   * @param logger The callback used to log messages to the LSP output channel.
   *     This logger is expected to append a newline after each invocation.
   * @param onExit A callback invoked whenever the server exits.
   */
  constructor({
    initializationOptions,
    logger,
    onExit,
    onNotification,
    onOutgoingRequest,
  }: {
    initializationOptions: InitializationOptions;
    logger: (message: string) => void;
    onExit: (status: ExitStatus) => void;
    onNotification: (method: string, params: JSONObject) => void;
    onOutgoingRequest: (id: any, method: string, params: JSONObject) => void;
  }) {
    super();

    this.serverProcess = spawn(
      initializationOptions.serverPath,
      initializationOptions.serverArgs,
      {
        env: initializationOptions.serverEnv,
      },
    );
    this.pushSubscription(
      new LineSeparatedStream(this.serverProcess.stderr!, (line: string) =>
        logger(line),
      ),
    );
    this.pushSubscription(
      new JSONRPCStream(
        this.serverProcess.stdout!,
        (response: JSONObject) =>
          this.pendingRequests.get(response.id)!.responseStream.next(response),
        (notification: JSONObject) =>
          onNotification(notification.method, notification.params),
        (request: JSONObject) =>
          onOutgoingRequest(request.id, request.method, request.params),
      ),
    );
    this.pushSubscription(new ProcessExitStream(this.serverProcess, onExit));
    this.pushSubscription(
      new DisposableCallback(() => {
        // We kill the server process after all listeners have been disposed, to
        // guarantee that no listener is invoked when the process dies.
        try {
          this.serverProcess.kill();
        } catch {}
      }),
    );
  }

  /**
   * Send a request to the server given its params and a method name that
   * follows the LSP protocol.
   * @returns a promise with the payload that gets resolved when the request is
   *     responded.
   */
  public async sendRequest(
    params: RequestParams,
    method: string,
  ): Promise<JSONObject> {
    const request = this.wrapRequest(params, method);
    const id = request.id;
    await this.sendPacket(request);

    const subject = new Subject<any>();
    this.pendingRequests.set(id, { params: params, responseStream: subject });
    const result = (await firstValueFrom(subject)).result;
    this.pendingRequests.delete(id);
    return result;
  }

  /**
   * Send a notification to the server given its params and a method name that
   * follows the LSP protocol.
   */
  public sendNotification<T>(params: T, method: string): void {
    const notification = this.wrapNotification(params, method);
    this.sendPacket(notification);
  }

  /**
   * Send a response to a server -> client request, given a response body and a
   * request ID.
   */
  public sendResponse(id: any, result: unknown): void {
    const response = this.wrapResponse(id, result);
    this.sendPacket(response);
  }

  public sendError(id: any, error: unknown): void {
    const response = this.wrapResponse(id, undefined, error);
    this.sendPacket(response);
  }

  /**
   * @returns A new incremental request Id that can be used for sending
   *     requests.
   */
  private getNewRequestId(): number {
    this.lastSentRequestId++;
    return this.lastSentRequestId;
  }

  /**
   *  Sends some arbitrary data that is sent to the server using the JSON RPC
   * protocol.
   */
  private async sendPacket<T>(packet: T): Promise<void> {
    const payload = Buffer.from(JSON.stringify(packet));
    return new Promise((resolve, _reject) => {
      return this.serverProcess.stdin?.write(
        `${protocolHeader}${payload.length}${protocolLineSeparator}${payload}`,
        () => resolve(),
      );
    });
  }

  /**
   * Wraps some params and method within a new object that is ready to be sent
   * to the server as a request.
   */
  private wrapRequest<T>(params: T, method: string): any {
    return {
      id: this.getNewRequestId(),
      jsonrpc: '2.0',
      method: method,
      params: params,
    };
  }

  /**
   * Wraps some params and method within a new object that is ready to be sent
   * to the server as a notification.
   */
  private wrapNotification<T>(params: T, method: string): any {
    return {
      jsonrpc: '2.0',
      method: method,
      params: params,
    };
  }

  /**
   * Wraps an ID and params as a response object,
   */
  private wrapResponse(id: any, result?: unknown, error?: unknown): JSONObject {
    return {
      jsonrpc: '2.0',
      id,
      result,
      error,
    };
  }

  /**
   * @returns the params of the oldest pending request.
   */
  public getOldestPendingRequest(): Optional<RequestParams> {
    for (const params of this.pendingRequests.values()) {
      return params.params;
    }
    return undefined;
  }
}
