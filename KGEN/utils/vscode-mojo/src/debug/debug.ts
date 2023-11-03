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

    let config = await this.context?.getSDK().resolveConfig(
        session.configuration.modularHomePath || session.workspaceFolder);
    if (!config)
      return null;
    // The --repl-mode set to `auto` indicates LLDB to distinguish automatically
    // if the text passed in the debug console is an expression or a command and
    // handle it accordingly. In case of ambiguity, the user can use the `:`
    // prefix to force it being a regular command, just like the REPL.
    return new vscode.DebugAdapterExecutable(config.mojoLLDBVSCodePath,
                                             [ "--repl-mode", "auto" ]);
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
    // The timeout that will be used by LLDB when initializing the target in
    // different scenarios. We use 5 minutes as a very conservative timeout when
    // debugging massive LLVM targets.
    const initializationTimeoutSec = 5 * 60;

    // This setting indicates LLDB to generate a useful summary for each
    // non-primitive type that is displayed right away in the IDE.
    if (!("enableAutoVariableSummaries" in debugConfiguration))
      debugConfiguration["enableAutoVariableSummaries"] = true;

    // This setting indicates LLDB to use the `:` prefix in the Debug Console to
    // disambiguate variable printing from regular LLDB commands.
    if (!("commandEscapePrefix" in debugConfiguration))
      debugConfiguration["commandEscapePrefix"] = ':';

    // This timeout affects targets created with "attachCommands" or
    // "launchCommands".
    if (!("timeout" in debugConfiguration))
      debugConfiguration["timeout"] = initializationTimeoutSec;

    // This setting shortens the length of address strings.
    const initCommands = [
      "settings set target.show-hex-variable-values-with-leading-zeroes false"
    ];

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

    const env = [
      `LLDB_VSCODE_RIT_TIMEOUT_IN_MS=${
          initializationTimeoutSec *
          1000}` // runInTerminal initialization timeout.
    ];

    // We add the MODULAR_HOME env var to enable debugging of SDK artifacts,
    // giving preference to the env specified by the user.
    if (config)
      env.push(`MODULAR_HOME=${config.modularHomePath}`);

    debugConfiguration["env"] = [...env, ...(debugConfiguration["env"] || []) ];
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
          // We fully restart all the servers after a workspace event for
          // simplicity.
          for (const [_, rpcServer] of this.rpcServers) {
            rpcServer.dispose();
          }
          this.rpcServers.clear();
          this.launchRpcServers();
        }));
    // Initialize the RPC servers.
    this.launchRpcServers();
  }

  private launchRpcServers(): void {
    // It's not possible to ask VS Code for the settings that are specific to a
    // given workspace or to the user. In fact, you can only provide some
    // "context" and then VS Code will return a set of settings that might come
    // from different places all merged together. Because of this, we need to
    // fetch settings from different contexts and reuse servers whenever
    // possible.
    for (const folder of vscode.workspace.workspaceFolders || []) {
      this.updateOrCreateRpcServer(folder);
    }
    this.updateOrCreateRpcServer();
  }

  /**
   * Create a debug rpc server using the config from the given workspace. If the
   * workspace is undefined, then a global config is used instead.
   */
  private updateOrCreateRpcServer(workspaceFolder?: vscode.WorkspaceFolder) {
    let options = config.get<{port?: number, token?: string}>('lldb.rpcServer',
                                                              workspaceFolder);
    if (!options || Object.keys(options).length == 0)
      return;
    const port = options.port;
    if (port === undefined) {
      this.context.getLoggingService().logInfo(
          `The 'port' key was not found in the mojo.lldb.rpcServer settings.`,
          options);
      return;
    }

    const key = `${port}`;
    const existingServer = this.rpcServers.get(key);
    if (existingServer) {
      existingServer.addServerToken(options.token);
    } else {
      let rpcServer = new RpcLaunchServer(this.context.getLoggingService(),
                                          port, options.token);
      this.context.getLoggingService().logInfo(`Starting RPC server for port:`,
                                               port);
      this.pushSubscription(rpcServer);
      rpcServer.listen();
      this.rpcServers.set(key, rpcServer);
    }
  }
}
