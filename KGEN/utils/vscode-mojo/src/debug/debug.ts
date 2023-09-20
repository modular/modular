//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';
import * as config from '../utils/config';
import {DisposableContext} from '../utils/disposableContext';

import {
  RpcLaunchServer,
  RpcLaunchServerOptions,
  UriLaunchServer
} from './externalDebugLauncher';

/**
 * This class defines a factory used to find the lldb-vscode binary to use
 * depending on the session configuration.
 */
class MojoDebugAdapterDescriptorFactory implements
    vscode.DebugAdapterDescriptorFactory {
  private context: MOJOContext|undefined;
  public static DEBUG_TYPE: string = "mojo-lldb";

  constructor(context: MOJOContext) { this.context = context; }

  async createDebugAdapterDescriptor(session: vscode.DebugSession,
                                     _executable: vscode.DebugAdapterExecutable|
                                     undefined):
      Promise<vscode.DebugAdapterDescriptor|null> {
    let config =
        await this.context?.getSDK().resolveConfig(session.workspaceFolder);
    if (!config)
      return null;
    return new vscode.DebugAdapterExecutable(config.mojoLLDBVSCodePath, []);
  }
}

/**
 * This class modifies the debug configuration right before the debug adapter is
 * launched. In other words, this is where we configure lldb-vscode.
 */
class MojoDebugConfigurationProvider implements
    vscode.DebugConfigurationProvider {
  private context: MOJOContext|undefined;
  public static DEBUG_TYPE: string = "mojo-lldb";

  constructor(context: MOJOContext) { this.context = context; }

  async resolveDebugConfiguration(folder: vscode.WorkspaceFolder|undefined,
                                  debugConfiguration: vscode.DebugConfiguration,
                                  token?: vscode.CancellationToken):
      Promise<vscode.DebugConfiguration> {
    // This setting indicates LLDB to generate a useful summary for each
    // non-primitive type that is displayed right away in the IDE.
    if (!("enableAutoVariableSummaries" in debugConfiguration))
      debugConfiguration["enableAutoVariableSummaries"] = true;

    // This setting shortens the length of address strings.
    const initCommands =
        [ "settings set target.show-hex-values-with-leading-zeroes false" ];

    // Load the MojoLLDB plugin.
    let config = await this.context?.getSDK().resolveConfig(folder);
    if (config && config.mojoLLDBPluginPath &&
        config.mojoLLDBPluginPath.length > 0) {
      initCommands.push(`plugin load '${config.mojoLLDBPluginPath}'`);
    }

    // We give preference to the init commands specified by the user.
    debugConfiguration["initCommands"] = [
      ...initCommands,
      ...(debugConfiguration["initCommands"] || []),
    ];

    // We add the MODULAR_HOME env var to enable debugging of SDK artifacts,
    // giving preference to the env specified by the user.
    if (config) {
      debugConfiguration["env"] = [
        `MODULAR_HOME=${config.modularHomePath}`,
        ...(debugConfiguration["env"] || [])
      ];
    }
    return debugConfiguration;
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

    this.pushSubscription(vscode.debug.onDidStartDebugSession(listener => {
      if (listener.configuration.type !=
          MojoDebugAdapterDescriptorFactory.DEBUG_TYPE)
        return;
      if (!listener.configuration.runInTerminal)
        vscode.commands.executeCommand("workbench.debug.action.focusRepl");
    }));

    this.pushSubscription(vscode.debug.registerDebugConfigurationProvider(
        MojoDebugAdapterDescriptorFactory.DEBUG_TYPE,
        new MojoDebugConfigurationProvider(context)));

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
    let options =
        config.get<RpcLaunchServerOptions>('lldb.rpcServer', workspaceFolder);
    if (!options || Object.keys(options).length == 0)
      return;

    let uri = workspaceFolder?.uri.toString() || "";
    if (workspaceFolder)
      this.context.getLoggingService().logInfo(
          `Starting RPC server for workspace '${uri}'`, options);
    else
      this.context.getLoggingService().logInfo(
          "Starting RPC server defined by global config", options);

    this.disposeRpcServer(workspaceFolder);
    let rpcServer = new RpcLaunchServer(this.context.getLoggingService(),
                                        workspaceFolder, options);
    this.pushSubscription(rpcServer);
    rpcServer.listen();
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
    rpcServer.dispose();
    this.rpcServers.delete(uri);
  }
}
