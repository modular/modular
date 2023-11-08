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
import {
  debug,
  DebugConfiguration,
  EventEmitter,
  Uri,
  UriHandler,
  window,
  workspace
} from "vscode";
import * as YAML from 'yaml';

import {LoggingService} from '../logging';
import {DisposableContext} from '../utils/disposableContext';

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
      this.loggingService.logInfo(`Handling uri: ${uri}`);
      let query = decodeURIComponent(uri.query);
      this.loggingService.logInfo(`Decoded query:\n${query}`);

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

        let env: {[key: string]: string;} = {};
        for (let frag of frags) {
          let pos = frag.indexOf('=');
          if (pos > 0)
            env[frag.substring(0, pos)] = frag.substring(pos + 1);
        }

        let args = stringArgv(cmdLine || '');
        let program = args.shift();
        let debugConfig: DebugConfiguration = {
          type : 'mojo-lldb',
          request : 'launch',
          name : '',
          program : program,
          args : args,
          env : env,
        };
        debugConfig.name = debugConfig.name || debugConfig.program;
        await debug.startDebugging(undefined, debugConfig);

      } else if (uri.path == '/debug/launch-config') {
        let debugConfig: DebugConfiguration = {
          type : 'mojo-lldb',
          request : 'launch',
          name : '',
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

/**
 * RPC-based debug launcher.
 *
 * It listens for network messages containing full JSON debug configurations and
 * launches them using lldb-vscode.
 */
export class RpcLaunchServer extends DisposableContext {
  private inner: net.Server;
  private port: number;
  private tokens = new Set<string>();
  private errorEmitter = new EventEmitter<Error>();
  private loggingService: LoggingService;

  /**
   * This constructor receives an optional token, which is expected to match the
   * `token` attribute from the incoming debug configuration requests as a
   * safety mechanism.
   */
  constructor(loggingService: LoggingService, port: number,
              token: string|undefined) {
    super();
    this.loggingService = loggingService;
    this.port = port;
    this.addServerToken(token);

    this.pushSubscription(this.errorEmitter.event((e: Error) => {
      this.loggingService.logError(
          "RPC Server error. You might need to restart VS Code to fix this issue.",
          e);
    }));

    this.inner = net.createServer({allowHalfOpen : true});
    this.inner.on('error', err => { this.errorEmitter.fire(err); });
    this.inner.on('connection', socket => {
      let request = '';
      socket.on('data', chunk => request += chunk);
      socket.on('end', () => {
        let response = this.processRequest(request);
        if (response instanceof Promise) {
          response.then(value => socket.end(value));
        } else {
          socket.end(response);
        }
      });
    });
  }

  /**
   * Adds the given token, if not undefined, to the list of tokens used for
   * basic security of this server.
   */
  addServerToken(token: string|undefined) {
    if (token !== undefined)
      this.tokens.add(token)
  }

  /**
   * Process a JSON debug configuration. It should contain a token field with
   * the same value as the one defined to create the RPC server.
   */
  async processRequest(request: string) {
    let debugConfig: DebugConfiguration = {
      type : 'mojo-lldb',
      request : 'launch',
      name : '',
    };
    Object.assign(debugConfig, YAML.parse(request));
    debugConfig.name = debugConfig.name || debugConfig.program;
    if (this.tokens.size != 0) {
      if (!this.tokens.has(debugConfig.token || ''))
        return 'Invalid token. This mojo.lldb.rpcServer expects a different `token` attribute for the launch configuration.';
      delete debugConfig.token;
    }
    try {
      let success = await debug.startDebugging(/*workspaceFolder=*/ undefined,
                                               debugConfig);
      return JSON.stringify({success : success});
    } catch (err) {
      return JSON.stringify({success : false, message : `${err}`});
    }
  };

  /**
   * Listens to messages using the provided network options.
   */
  public async listen() {
    return new Promise<net.AddressInfo|string>(
        resolve =>
            this.inner.listen({port : this.port, host : "127.0.0.1"},
                              () => resolve(this.inner.address() || "")));
  }

  /**
   * Closes and disposes the server.
   */
  public dispose() {
    this.inner.close();
    super.dispose();
  }
}
