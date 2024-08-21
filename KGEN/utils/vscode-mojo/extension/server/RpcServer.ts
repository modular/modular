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
import {
  debug,
  DebugConfiguration,
} from 'vscode';
import { LoggingService } from '../logging';
import { DisposableContext } from '../utils/disposableContext';
import path = require('path');

type ResponseConnect = {
  kind: "connect";
  pid: number;
  lastTimeSeenActiveInSecs: number;
  name: string | undefined;
};

type ResponseDebug = {
  kind: "debug";
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
  kind: "connect";
};
type RequestDebug = {
  kind: "debug";
  debugConfiguration: DebugConfiguration;
}

function instanceOfConnect(object: any): object is RequestConnect {
  return object.kind === "connect";
}

function instanceOfDebug(object: any): object is RequestDebug {
  return object.kind === "debug" && typeof object.debugConfiguration === 'object';
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
  private loggingService: LoggingService;
  private readonly protocolSeparator = "\n----\n";
  private lastTimeActiveInMillis: Date = new Date();

  constructor(loggingService: LoggingService) {
    super();
    this.loggingService = loggingService;

    this.server = net.createServer({ allowHalfOpen: true });
    let clients: net.Socket[] = [];
    this.server.on('error', err => this.onError(err));
    this.server.on('connection', socket => {
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
          this.loggingService.main.logInfo("RPC server closed.");
          this.server.unref();
        });
      })
    );
    this.pushSubscription(vscode.window.onDidChangeWindowState((e: vscode.WindowState) => {
      if (e.active) {
        this.lastTimeActiveInMillis = new Date();
      }
    }));
  }

  private onError(err: Error): void {
    if (err.message.includes('EADDRINUSE') && this.port < PORT_MAX) {
      this.loggingService.main.logInfo("Will try to start the RPC Server with a new port.");
      this.port += 1;
      this.listen();
    } else {
      this.loggingService.main.logError(
        'RPC Server error. You might need to restart VS Code to fix this issue.',
        err
      );
    }
  }

  // Launch a debug session. Throws if debug session initialization has error.
  private async handleDebugRequest(debugConfig: DebugConfiguration) {
    debugConfig.name = debugConfig.name || debugConfig.program;
    if (debugConfig.type == "cuda-gdb") {
      const nsight =
        vscode.extensions.getExtension("nvidia.nsight-vscode-edition");
      if (!nsight) {
        // Tell the user to install the nsight extension.
        const message = "Unable to start the cuda-gdb debug session. You first need to install the NVIDIA Nsight extension (nsight-vscode-edition)";
        this.loggingService.main.logInfo(message);
        const response = await vscode.window.showInformationMessage("Unable to debug in cuda-gdb mode without NVIDIA Nsight extension.",
                                                                    "Find NVIDIA Nsight extension");
        if (response) {
          vscode.commands.executeCommand("workbench.extensions.search", "@id:nvidia.nsight-vscode-edition");
        }
        throw new Error(message);
      }
    }
    let success = await debug.startDebugging(
      /*workspaceFolder=*/ undefined,
      debugConfig
    );
    if (!success)
      throw new Error("Unable to start the debug session");
  }

  private async dispatchRequest(socket: net.Socket, rawRequest: string): Promise<void> {
    let request: any | undefined;
    try {
      const parsedRequest = JSON.parse(rawRequest);
      if (typeof parsedRequest === 'object') {
        request = parsedRequest;
      }
    } catch (err) {
      this.loggingService.main.logInfo(`RPC Server request parsing error: ${err}`);
    }

    if (request === undefined) {
      const response: RPCServerResponse = {
        success: false,
        message: 'Malformed request. Not a JSON object.',
      };
      socket.end(JSON.stringify(response) + this.protocolSeparator);
      return;
    }
    this.loggingService.main.logInfo(`RPC Server request: ${JSON.stringify(request)}`);

    if (instanceOfConnect(request)) {
      let name = "[VSCode]";
      const activeTab = vscode.window.tabGroups.activeTabGroup.activeTab?.label;
      if (activeTab !== undefined) {
        name += ` ${activeTab}`;
      } else {
        const ws = vscode.workspace.workspaceFolders?.at(0);
        if (ws !== undefined) {
          name += ` ${ws.name}`;
        }
      }

      const now = (new Date()).getTime();
      const response: RPCServerResponse = {
        success: true,
        kind: "connect",
        pid: process.pid,
        lastTimeSeenActiveInSecs: Math.floor((now - this.lastTimeActiveInMillis.getTime()) / 1000),
        name,
      };
      this.loggingService.main.logInfo(`RPC Server response: ${JSON.stringify(response)}`);
      socket.write(JSON.stringify(response) + this.protocolSeparator);
    } else if (instanceOfDebug(request)) {
      const debugConfig: DebugConfiguration = request.debugConfiguration;
      try {
        await this.handleDebugRequest(debugConfig);
        const response: RPCServerResponse = {
          success: true,
          kind: "debug",
        };
        this.loggingService.main.logInfo(`RPC Server response: ${JSON.stringify(response)}`);
        socket.write(JSON.stringify(response) + this.protocolSeparator);
      } catch (err) {
        const response: RPCServerResponse = {
          success: false,
          message: `${err}`,
          kind: "debug"
        };
        this.loggingService.main.logInfo(`RPC Server response: ${JSON.stringify(response)}`);
        socket.write(JSON.stringify(response) + this.protocolSeparator);
      }
    } else {
      const response: RPCServerResponse = {
        success: false,
        message: 'Invalid request',
      };
      this.loggingService.main.logInfo(`RPC Server response: ${JSON.stringify(response)}`);
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
    this.loggingService.main.logInfo(`Attempting to create the RPC server with port ${this.port}`);

    return new Promise<net.AddressInfo | string>((resolve) =>
      this.server.listen({ port: this.port, host: '127.0.0.1' }, () =>
        resolve(this.server.address() || '')
      )
    );
  }
}
