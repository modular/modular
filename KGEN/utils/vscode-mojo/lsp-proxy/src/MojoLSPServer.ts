//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {ChildProcess, spawn} from 'child_process';
import {firstValueFrom, Subject} from 'rxjs';

import {JSONRPCStream, LineSeparatedStream} from './streams';
import {ExitStatus, InitializationOptions} from './types';

type JSONObject = {
  [key: string]: any
};
type RequestId = number;

const protocolHeader = "Content-Length: ";
const protocolLineSeparator = "\r\n\r\n";

/**
 * This class manages an instance of the mojo-lsp-server process, as well as
 * supporting utilities for sending requests and notifications.
 */
export class MojoLSPServer {
  private serverProcess: ChildProcess;
  // Request response tracker.
  private lastRequestId: RequestId = -1;
  private pendingRequests = new Map<RequestId, Subject<JSONObject>>();
  private exitStream = new Subject<ExitStatus>();

  /**
   * @param initializationOptions The options needed to spawn the
   *     mojo-lsp-server.
   * @param logger The callback used to log messages to the LSP output channel.
   *     This logger is expected to append a newline after each invocation.
   * @param onExit A callback invoked whenever the server exits.
   */
  constructor({initializationOptions, logger, onExit, onNotification}: {
    initializationOptions: InitializationOptions;
    logger : (message: string) => void;
    onExit : (status: ExitStatus) => void;
    onNotification : (method: string, params: JSONObject) => void;
  }) {
    this.exitStream.subscribe(onExit);

    this.serverProcess = spawn(initializationOptions.serverPath,
                               initializationOptions.serverArgs, {
                                 env : initializationOptions.serverEnv,
                               });
    new LineSeparatedStream(this.serverProcess.stderr!,
                            (line: string) => logger(line));
    new JSONRPCStream(this.serverProcess.stdout!,
                      (request: JSONObject) => {
                        const subject = this.pendingRequests.get(request.id)!;
                        subject.next(request);
                      },
                      (notification: JSONObject) => onNotification(
                          notification.method, notification.params));
    this.setupServerExit(logger);
  }

  /**
   * Send a request to the server given its params and a method name that
   * follows the LSP protocol.
   * @returns a promise with the payload that gets resolved when the request is
   *     responded.
   */
  public async sendRequest<T>(params: T, method: string): Promise<any> {
    const request = this.wrapRequest(params, method);
    const id = request.id;
    await this.sendPacket(request);

    const subject = new Subject<any>();
    this.pendingRequests.set(id, subject);
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
   * @returns A new incremental request Id that can be used for sending
   *     requests.
   */
  private getNewRequestId(): number {
    this.lastRequestId++;
    return this.lastRequestId;
  }

  /**
   *  Sends some arbitrary data that is sent to the server using the JSON RPC
   * protocol.
   */
  private async sendPacket<T>(packet: T): Promise<void> {
    const payload = Buffer.from(JSON.stringify(packet));
    return new Promise((resolve, _reject) => {
      return this.serverProcess.stdin?.write(
          `${protocolHeader}${payload.length}${protocolLineSeparator}${
              payload}`,
          () => resolve());
    });
  }

  /**
   * Wraps some params and method within a new object that is ready to be sent
   * to the server as a request.
   */
  private wrapRequest<T>(params: T, method: string): any {
    return {
      id : this.getNewRequestId(),
      jsonrpc : "2.0",
      method : method,
      params : params,
    };
  }

  /**
   * Wraps some params and method within a new object that is ready to be sent
   * to the server as a notification.
   */
  private wrapNotification<T>(params: T, method: string): any {
    return {
      jsonrpc : "2.0",
      method : method,
      params : params,
    };
  }

  private setupServerExit(logger: any) {
    this.serverProcess.on(
        "exit",
        (code: number|null, signal: NodeJS.Signals|
                            null) => { this.exitStream.next({code, signal}); });
  }
}
