//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// The following code is a modification of
// https://github.com/vadimcn/codelldb/blob/master/extension/externalLaunch.ts,
// which has MIT license.

import * as net from 'net';
import * as vscode from 'vscode';
import { debug, DebugConfiguration } from 'vscode';
import { Logger } from '../logging';
import { checkNsightInstall } from '../utils/checkNsight';
import { DisposableContext } from '../utils/disposableContext';

type ResponseConnect = {
  kind: 'connect';
  pid: number;
  lastTimeSeenActiveInSecs: number;
  name: Optional<string>;
};

type ResponseDebug = {
  kind: 'debug';
};
type Response = ResponseConnect | ResponseDebug;

type RPCServerResponse =
  | ({ success: true } & Response)
  | {
      success: false;
      message?: string;
      kind?: string;
    };

type RequestConnect = {
  kind: 'connect';
};
type RequestDebug = {
  kind: 'debug';
  debugConfiguration: DebugConfiguration;
};

function instanceOfConnect(object: any): object is RequestConnect {
  return object.kind === 'connect';
}

function instanceOfDebug(object: any): object is RequestDebug {
  return (
    object.kind === 'debug' && typeof object.debugConfiguration === 'object'
  );
}

const PORT_MIN = 12355;
const PORT_MAX = 12364; // Inclusive

/**
 * RPC Server.
 *
 * It listens for network messages dispatching actions on this extension.
 * Messages are JSON objects followed by a `\n----\n`.
 */
export class RpcServer extends DisposableContext {
  private server: net.Server;
  private port: number = PORT_MIN;
  private logger: Logger;
  private readonly protocolSeparator = '\n----\n';
  private lastTimeActiveInMillis: Date = new Date();

  constructor(logger: Logger) {
    super();
    this.logger = logger;

    this.server = net.createServer({ allowHalfOpen: true });
    let clients: net.Socket[] = [];
    this.server.on('error', (err) => this.onError(err));
    this.server.on('connection', (socket) => {
      this.configureSocket(socket);
      clients.push(socket);
      socket.on('close', () => {
        clients.splice(clients.indexOf(socket), 1);
      });
    });

    this.pushSubscription(
      new vscode.Disposable(() => {
        for (const client of clients) {
          client.destroy();
        }
        this.server.close(() => {
          this.logger.info('RPC server closed.');
          this.server.unref();
        });
      }),
    );
    this.pushSubscription(
      vscode.window.onDidChangeWindowState((e: vscode.WindowState) => {
        if (e.active) {
          this.lastTimeActiveInMillis = new Date();
        }
      }),
    );
  }

  private onError(err: Error): void {
    if (err.message.includes('EADDRINUSE') && this.port < PORT_MAX) {
      this.logger.info('Will try to start the RPC Server with a new port.');
      this.port += 1;
      this.listen();
    } else {
      this.logger.error(
        'RPC Server error. You might need to restart VS Code to fix this issue.',
        err,
      );
    }
  }

  // Launch a debug session. Throws if debug session initialization has error.
  private async handleDebugRequest(debugConfig: DebugConfiguration) {
    debugConfig.name = debugConfig.name || debugConfig.program;
    if (debugConfig.type === 'mojo-cuda-gdb') {
      const maybeErrorMessage = await checkNsightInstall(this.logger);
      if (maybeErrorMessage) {
        throw new Error(maybeErrorMessage);
      }
    }
    let success = await debug.startDebugging(
      /*workspaceFolder=*/ undefined,
      debugConfig,
    );

    if (!success) {
      throw new Error('Unable to start the debug session');
    }
  }

  private async dispatchRequest(
    socket: net.Socket,
    rawRequest: string,
  ): Promise<void> {
    let request: Optional<any>;
    try {
      const parsedRequest = JSON.parse(rawRequest);
      if (typeof parsedRequest === 'object') {
        request = parsedRequest;
      }
    } catch (err) {
      this.logger.info(`RPC Server request parsing error: ${err}`);
    }

    if (request === undefined) {
      const response: RPCServerResponse = {
        success: false,
        message: 'Malformed request. Not a JSON object.',
      };
      socket.end(JSON.stringify(response) + this.protocolSeparator);
      return;
    }
    this.logger.info(`RPC Server request: ${JSON.stringify(request)}`);

    if (instanceOfConnect(request)) {
      let name = '[VSCode]';
      const activeTab = vscode.window.tabGroups.activeTabGroup.activeTab?.label;
      if (activeTab !== undefined) {
        name += ` ${activeTab}`;
      } else {
        const ws = vscode.workspace.workspaceFolders?.at(0);
        if (ws !== undefined) {
          name += ` ${ws.name}`;
        }
      }

      const now = new Date().getTime();
      const response: RPCServerResponse = {
        success: true,
        kind: 'connect',
        pid: process.pid,
        lastTimeSeenActiveInSecs: Math.floor(
          (now - this.lastTimeActiveInMillis.getTime()) / 1000,
        ),
        name,
      };
      this.logger.info(`RPC Server response: ${JSON.stringify(response)}`);
      socket.write(JSON.stringify(response) + this.protocolSeparator);
    } else if (instanceOfDebug(request)) {
      const debugConfig: DebugConfiguration = request.debugConfiguration;
      try {
        await this.handleDebugRequest(debugConfig);
        const response: RPCServerResponse = {
          success: true,
          kind: 'debug',
        };
        this.logger.info(`RPC Server response: ${JSON.stringify(response)}`);
        socket.write(JSON.stringify(response) + this.protocolSeparator);
      } catch (err) {
        const response: RPCServerResponse = {
          success: false,
          message: `${err}`,
          kind: 'debug',
        };
        this.logger.info(`RPC Server response: ${JSON.stringify(response)}`);
        socket.write(JSON.stringify(response) + this.protocolSeparator);
      }
    } else {
      const response: RPCServerResponse = {
        success: false,
        message: 'Invalid request',
      };
      this.logger.info(`RPC Server response: ${JSON.stringify(response)}`);
      socket.end(JSON.stringify(response) + this.protocolSeparator);
    }
  }

  private configureSocket(socket: net.Socket) {
    let buffer = '';
    socket.on('data', async (chunk: any) => {
      buffer += chunk;
      while (buffer.includes(this.protocolSeparator)) {
        const pos = buffer.indexOf(this.protocolSeparator);
        const rawRequest = buffer.substring(0, pos);
        buffer = buffer.substring(pos + this.protocolSeparator.length);
        await this.dispatchRequest(socket, rawRequest);
      }
    });
    socket.on('end', () => {
      socket.end();
    });
  }

  /**
   * Listens to messages using the provided network options.
   */
  public async listen() {
    this.logger.info(
      `Attempting to create the RPC server with port ${this.port}`,
    );

    return new Promise<net.AddressInfo | string>((resolve) =>
      this.server.listen({ port: this.port, host: '127.0.0.1' }, () =>
        resolve(this.server.address() || ''),
      ),
    );
  }
}
