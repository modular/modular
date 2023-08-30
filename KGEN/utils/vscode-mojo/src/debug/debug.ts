//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';
import * as config from '../utils/config';
import {DisposableContext} from '../utils/disposableContext';

import {RpcLaunchServer, UriLaunchServer} from './externalDebugLauncher';

/**
 * This class defines a factory used to create the debug adaptor used for
 * debugging mojo, which is dependent on the SDK installation.
 */
class MojoDebugAdapterDescriptorFactory implements
    vscode.DebugAdapterDescriptorFactory {
  _context: MOJOContext|undefined;
  public static DEBUG_TYPE: string = "mojo-lldb";

  constructor(context: MOJOContext) { this._context = context; }

  async createDebugAdapterDescriptor(session: vscode.DebugSession,
                                     executable: vscode.DebugAdapterExecutable|
                                     undefined):
      Promise<vscode.DebugAdapterDescriptor|null> {
    let config =
        await this._context?.getSDK().resolveConfig(session.workspaceFolder);
    if (!config)
      return null;
    return new vscode.DebugAdapterExecutable(config.mojoLLDBVSCodePath, []);
  }
}

/**
 * Class used to register and manage all the necessary constructs to support
 * mojo debugging.
 */
export class MojoDebugContext extends DisposableContext {
  private context: MOJOContext;
  rpcServers: Map<string, RpcLaunchServer> = new Map();

  constructor(context: MOJOContext) {
    super();
    this.context = context;

    // Register the lldb-vscode debug adapter.
    this.pushSubscription(vscode.debug.registerDebugAdapterDescriptorFactory(
        MojoDebugAdapterDescriptorFactory.DEBUG_TYPE,
        new MojoDebugAdapterDescriptorFactory(context)));

    // Register the URI-based debug launcher.
    this.pushSubscription(vscode.window.registerUriHandler(
        new UriLaunchServer(context.getLoggingService())));

    // Register the RPC-based debug launcher.
    this.pushSubscription(
        vscode.workspace.onDidChangeWorkspaceFolders((event) => {
          for (const folder of event.removed) {
            this.disposeRpcServer(folder);
          }
          for (const folder of event.added) {
            this.updateOrCreateRpcServer(folder);
          }
        }));

    // Initialize the RPC server.
    this.updateOrCreateRpcServer();
    for (const folder of vscode.workspace.workspaceFolders || []) {
      this.updateOrCreateRpcServer(folder);
    }
  }

  /**
   * Create a debug rpc server using the config from the given workspace. If the
   * workspace is undefined, then a global config is used instead.
   */
  private updateOrCreateRpcServer(workspaceFolder?: vscode.WorkspaceFolder) {
    let options = config.get<any>('lldb.rpcServer', workspaceFolder);
    if (!options)
      return;

    let uri = workspaceFolder?.uri.toString() || "";
    if (workspaceFolder)
      this.context.getLoggingService().logInfo(
          `Starting RPC server for workspace '${uri}'`, options);
    else
      this.context.getLoggingService().logInfo(
          "Starting RPC server defined by global config", options);

    this.disposeRpcServer(workspaceFolder);
    let rpcServer = new RpcLaunchServer({token : options.token});
    rpcServer.listen(options);
    this.rpcServers.set(uri, rpcServer);
  }

  /**
   * Dispose the debug RPC server that was created by the given workspace
   * folder. If the workspace is undefined, then the global server is disposed
   * instead.
   */
  private disposeRpcServer(workspaceFolder: vscode.WorkspaceFolder|undefined) {
    let uri = workspaceFolder?.uri.toString() || "";
    let rpcServer = this.rpcServers.get(uri);
    if (!rpcServer)
      return;

    if (workspaceFolder) {
      this.context.getLoggingService().logInfo(
          `Stopping RPC server for workspace '${uri}'`);
    } else {
      this.context.getLoggingService().logInfo(
          `Stopping RPC server defined by global config`);
    }
    rpcServer.close();
    this.rpcServers.delete(uri);
  }
}
