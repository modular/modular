//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// The following code is a modification of
// https://github.com/vadimcn/codelldb/blob/master/extension/externalLaunch.ts,
// which has MIT license.

import * as net from 'net';
import * as querystring from 'querystring';
import stringArgv from 'string-argv';
import * as vscode from 'vscode';
import {
  debug,
  DebugConfiguration,
  EventEmitter,
  Uri,
  UriHandler,
  window,
  workspace,
} from 'vscode';
import * as YAML from 'yaml';

import { LoggingService } from '../logging';
import { DisposableContext } from '../utils/disposableContext';

type RPCServerResponse =
  | {
      success: true;
    }
  | {
      success: false;
      message?: string;
    };

/**
 * URI-based debug launcher.
 *
 * This handled VSCode URI requests of the form
 *
 * vscode://modular.vscode-mojo/debug?name=<configuration_name>,[folder=<path>]
 *
 * In this case, `<configuration name>` is the name of a debug configuration
 * defined in the the workspace `folder`, which might me undefined for a global
 * configuration.
 *
 * vscode://modular.vscode-mojo/debug/launch?<env1>=<val1>&<env2>=<val2>&<command-line>
 *
 * This launches the program specified by the given environment variables and
 * command line arguments.
 *
 * vscode://modular.vscode-mojo/debug/launch-config?<yaml>
 *
 * This starts a launch debug session given the <yaml> encoded debug
 * configuration.
 */
export class UriLaunchServer implements UriHandler {
  private loggingService: LoggingService;

  constructor(loggingService: LoggingService) {
    this.loggingService = loggingService;
  }

  async handleUri(uri: Uri) {
    try {
      this.loggingService.main.logInfo(`Handling uri: ${uri}`);
      let query = decodeURIComponent(uri.query);
      this.loggingService.main.logInfo(`Decoded query:\n${query}`);

      if (uri.path == '/debug') {
        let params = querystring.parse(uri.query, ',') as {
          [key: string]: string;
        };
        if (params.folder && params.name) {
          let wsFolder = workspace.getWorkspaceFolder(Uri.file(params.folder));
          await debug.startDebugging(wsFolder, params.name);
        } else if (params.name) {
          await debug.startDebugging(/*folder=*/ undefined, params.name);
        } else {
          throw new Error(`Unsupported combination of launch Uri parameters.`);
        }
      } else if (uri.path == '/debug/launch') {
        let frags = query.split('&');
        let cmdLine = frags.pop();

        let env: { [key: string]: string } = {};
        for (let frag of frags) {
          let pos = frag.indexOf('=');

          if (pos > 0) {
            env[frag.substring(0, pos)] = frag.substring(pos + 1);
          }
        }

        let args = stringArgv(cmdLine || '');
        let program = args.shift();
        let debugConfig: DebugConfiguration = {
          type: 'mojo-lldb',
          request: 'launch',
          name: '',
          program: program,
          args: args,
          env: env,
        };
        debugConfig.name = debugConfig.name || debugConfig.program;
        await debug.startDebugging(undefined, debugConfig);
      } else if (uri.path == '/debug/launch-config') {
        let debugConfig: DebugConfiguration = {
          type: 'mojo-lldb',
          request: 'launch',
          name: '',
        };
        Object.assign(debugConfig, YAML.parse(query));
        debugConfig.name = debugConfig.name || debugConfig.program;
        await debug.startDebugging(/*folder=*/ undefined, debugConfig);
      } else {
        throw new Error(`Unsupported Uri path: ${uri.path}`);
      }
    } catch (err) {
      await window.showErrorMessage(`${err}`);
    }
  }
}

const PORT_MIN = 12355;
const PORT_MAX = 12364; // Inclusive

/**
 * RPC-based debug launcher.
 *
 * It listens for network messages containing full JSON debug configurations and
 * launches them using lldb-vscode.
 */
export class RpcLaunchServer extends DisposableContext {
  private server: net.Server;
  private port: number | undefined = PORT_MIN;
  private errorEmitter = new EventEmitter<Error>();
  private loggingService: LoggingService;

  /**
   * This constructor receives an secret, which is expected to match the
   * `secret` attribute from the incoming debug configuration requests as a
   * safety mechanism.
   */
  constructor(loggingService: LoggingService) {
    super();
    this.loggingService = loggingService;

    this.pushSubscription(
      this.errorEmitter.event((e: Error) => {
        this.loggingService.main.logError(
          'RPC Server error. You might need to restart VS Code to fix this issue.',
          e
        );
      })
    );

    this.server = net.createServer({ allowHalfOpen: true });
    this.server.on('error', (err) => {
      this.errorEmitter.fire(err);
      if (err.message.includes('EADDRINUSE')) {
        if (this.port !== undefined && this.port < PORT_MAX) {
          this.loggingService.main.logInfo("Will try to start the RPC Server with a new port.")
          this.port += 1;
          this.listen();
        }
      } 
    });
    this.server.on('connection', (socket) => {
      let request = '';
      socket.on('data', (chunk) => {
        request += chunk;
        let parsedRequest: Object | undefined = undefined;
        try {
          parsedRequest = JSON.parse(request);
        } catch (err) {
          parsedRequest = undefined;
          // If we get an exception, parsedRequest will be undefined,
          // which is a case we'll handle below.
        }

        if (typeof parsedRequest === 'object') {
          this.processRequest(parsedRequest).then((value) =>
            socket.end(JSON.stringify(value))
          );
        } else if (parsedRequest !== undefined) {
          const response: RPCServerResponse = {
            success: false,
            message: 'the debug session request is not a JSON object.',
          };
          socket.end(JSON.stringify(response));
        }
        // In we couldn't parse, i.e. parsedRequest is undefined, it might be
        // because the data is incomplete, so we keep reading.
      });
      socket.on('end', () => {
        // If we got here, we check if we had an syntax error and return it as a
        // message.
        try {
          const _ = JSON.parse(request);
        } catch (err) {
          const response: RPCServerResponse = {
            success: false,
            message: `${err}`,
          };
          socket.end(JSON.stringify(response));
        }
        socket.end();
      });
    });

    this.pushSubscription(
      new vscode.Disposable(() => {
        this.server.close();
      })
    );
  }

  /**
   * Process a JSON debug configuration. It should contain a secret field with
   * the same value as the one defined to create the RPC server.
   */
  async processRequest(request: Object): Promise<RPCServerResponse> {
    this.loggingService.main.logInfo('Received RPC debug request', request);
    let debugConfig: DebugConfiguration = {
      type: 'mojo-lldb',
      request: 'launch',
      name: '',
    };
    Object.assign(debugConfig, request);
    debugConfig.name = debugConfig.name || debugConfig.program;
    try {
      let success = await debug.startDebugging(
        /*workspaceFolder=*/ undefined,
        debugConfig
      );
      return { success: success };
    } catch (err) {
      return { success: false, message: `${err}` };
    }
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
