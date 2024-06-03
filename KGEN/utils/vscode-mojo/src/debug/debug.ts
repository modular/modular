//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MojoContext} from '../mojoContext';
import * as config from '../utils/config';
import {DisposableContext} from '../utils/disposableContext';
import {getAllOpenMojoFiles, WorkspaceAwareFile} from '../utils/files';

import {activatePickProcessToAttachCommand} from './attachQuickPick';
import {RpcLaunchServer, UriLaunchServer} from './externalDebugLauncher';
import {
  initializeInlineLocalVariablesProvider,
} from './inlineVariables';

/**
 * Stricter version of vscode.DebugConfiguration intended to reduce the chances
 * of typos when handling individual attributes.
 */
type MojoDebugConfiguration = {
  type?: string;
  name?: string;
  pid?: string | number;
  request?: string;
  modularHomePath?: string;
  args?: string[];
  program?: string;
  mojoFile?: string;
  env?: string[];
  enableAutoVariableSummaries?: boolean;
  commandEscapePrefix?: string;
  timeout?: number;
  initCommands?: string[];
  customFrameFormat?: string;
}

/**
 * The "type" for debug configurations.
 */
const DEBUG_TYPE: string = "mojo-lldb";

/**
 * This class defines a factory used to find the lldb-vscode binary to use
 * depending on the session configuration.
 */
class MojoDebugAdapterDescriptorFactory implements
    vscode.DebugAdapterDescriptorFactory {
  private context: MojoContext;

  constructor(context: MojoContext) { this.context = context; }

  async createDebugAdapterDescriptor(session: vscode.DebugSession,
                                     _executable: vscode.DebugAdapterExecutable|
                                     undefined):
      Promise<vscode.DebugAdapterDescriptor|undefined> {
    let sdk = await this.context.sdkManager.findSDK();
    // We don't need to show error messages here because
    // `findSDKConfigForDebugSession` does that.
    if (!sdk)
      return undefined;
    return new vscode.DebugAdapterExecutable(sdk.config.mojoLLDBVSCodePath,
                                             [ "--repl-mode", "variable" ]);
  }
}

/**
 * This class modifies the debug configuration right before the debug adapter is
 * launched. In other words, this is where we configure lldb-vscode.
 */
class MojoDebugConfigurationResolver implements
    vscode.DebugConfigurationProvider {
  private context: MojoContext;

  constructor(context: MojoContext) { this.context = context; }

  async resolveDebugConfigurationWithSubstitutedVariables
      ?(folder: vscode.WorkspaceFolder|undefined,
        debugConfiguration: MojoDebugConfiguration,
        token?: vscode.CancellationToken):
          Promise<undefined|vscode.DebugConfiguration> {
    // Load the MojoLLDB plugin. The SDK must be present because otherwise we
    // can't get access to the debug adapter.
    let sdk = await this.context.sdkManager.findSDK();
    // We don't need to show error messages here because
    // `findSDKConfigForDebugSession` does that.
    if (!sdk)
      return undefined;

    if (typeof (debugConfiguration.pid) === "string") {
      debugConfiguration.pid = parseInt(debugConfiguration.pid);
    }

    if (debugConfiguration.mojoFile) {
      if (!debugConfiguration.mojoFile.endsWith('.🔥') &&
          !debugConfiguration.mojoFile.endsWith('.mojo')) {
        const message = `Mojo Debug error: the file '${
            debugConfiguration
                .mojoFile}' doesn't have the .🔥 or .mojo extension.`;
        this.context.loggingService.main.logError(message);
        vscode.window.showErrorMessage(message);
        return undefined;
      }
      debugConfiguration.args = [
        "run", "--no-optimization", "--debug-level", "full",
        debugConfiguration.mojoFile, ...(debugConfiguration.args || [])
      ];
      debugConfiguration.program = sdk.config.mojoDriverPath;
    }

    // We give preference to the init commands specified by the user.
    // The timeout that will be used by LLDB when initializing the target in
    // different scenarios. We use 5 minutes as a very conservative timeout when
    // debugging massive LLVM targets.
    const initializationTimeoutSec = 5 * 60;

    if (debugConfiguration.customFrameFormat === undefined) {
      // FIXME(#23274): include {${function.is-optimized} [opt]} when we don't
      // emit opt for -O0.
      debugConfiguration.customFrameFormat =
          "${function.name-with-args}{${frame.is-artificial} [artificial]}";
    }

    // This setting indicates LLDB to generate a useful summary for each
    // non-primitive type that is displayed right away in the IDE.
    if (debugConfiguration.enableAutoVariableSummaries === undefined)
      debugConfiguration.enableAutoVariableSummaries = true;

    // This setting indicates LLDB to use the `:` prefix in the Debug Console to
    // disambiguate variable printing from regular LLDB commands.
    if (debugConfiguration.commandEscapePrefix === undefined)
      debugConfiguration.commandEscapePrefix = ':';

    // This timeout affects targets created with "attachCommands" or
    // "launchCommands".
    if (debugConfiguration.timeout === undefined)
      debugConfiguration.timeout = initializationTimeoutSec;

    // This setting shortens the length of address strings.
    const initCommands = [
      "?settings set target.show-hex-variable-values-with-leading-zeroes false",
      // FIXME(#23274): remove this when we properly emit the opt flag.
      "?settings set target.process.optimization-warnings false",
      `?!plugin load '${sdk.config.mojoLLDBPluginPath}'`,
      "?mojo statistics telemetry session.start vscode"
    ];

    debugConfiguration.initCommands = [
      ...initCommands,
      ...(debugConfiguration.initCommands || []),
    ];

    // Pull in the additional visualizers within the lldb-visualizers dir.
    if (await sdk.config.lldbHasPythonScriptingSupport()) {
      let visualizersDir = sdk.config.mojoLLDBVisualizersPath;
      let visualizers = await vscode.workspace.fs.readDirectory(
          vscode.Uri.file(visualizersDir));
      let visualizerCommands = visualizers.map(
          ([ name, _type ]) =>
              `?command script import ${visualizersDir}/${name}`);
      debugConfiguration.initCommands.push(...visualizerCommands);
    }

    const env = [
      `LLDB_VSCODE_RIT_TIMEOUT_IN_MS=${
          initializationTimeoutSec *
          1000}` // runInTerminal initialization timeout.
    ];

    // We add the MODULAR_HOME env var to enable debugging of SDK artifacts,
    // giving preference to the env specified by the user.
    if (sdk.config.modularHomePath)
      env.push(`MODULAR_HOME=${sdk.config.modularHomePath}`);

    debugConfiguration.env = [...env, ...(debugConfiguration.env || []) ];
    return debugConfiguration as vscode.DebugConfiguration;
  }

  async resolveDebugConfiguration(folder: vscode.WorkspaceFolder|undefined,
                                  debugConfiguration: MojoDebugConfiguration,
                                  token?: vscode.CancellationToken):
      Promise<vscode.DebugConfiguration> {
    // The `Debug: Start Debugging` command (aka F5 or the `Run and Debug`
    // button if no launch.json files are present), invoke this method with a
    // totally empty debugConfiguration, so we have to fill it in.
    if (!debugConfiguration.request) {
      debugConfiguration.type = DEBUG_TYPE;
      debugConfiguration.request = "launch";
      // This will get replaced with the currently active document.
      debugConfiguration.mojoFile = "${file}";
    }

    return debugConfiguration as vscode.DebugConfiguration;
  }
}

/**
 * Provides debug configurations dynamically depending on the currently open
 * workspaces and files.
 */
class MojoDebugDynamicConfigurationProvider implements
    vscode.DebugConfigurationProvider {
  async provideDebugConfigurations(
      _folder: vscode.WorkspaceFolder|undefined,
      _token?: vscode.CancellationToken|undefined,
      ): Promise<vscode.DebugConfiguration[]|undefined> {
    const [activeFile, otherOpenFiles] = getAllOpenMojoFiles();
    return [ activeFile, ...otherOpenFiles ]
        .filter((file): file is WorkspaceAwareFile => !!file)
        .map(file => {
          return {
            type : DEBUG_TYPE,
            request : "launch",
            name : `Mojo: Debug ${file.baseName} ⸱ ${file.relativePath}`,
            mojoFile : file.uri.fsPath,
            args : [],
            env : [],
            cwd : file.workspaceFolder?.uri.fsPath,
            runInTerminal : false,
          };
        });
  }
}

/**
 * Class used to register and manage all the necessary constructs to support
 * mojo debugging.
 */
export class MojoDebugContext extends DisposableContext {
  private context: MojoContext;
  rpcServers: Map<string, RpcLaunchServer> = new Map();

  constructor(context: MojoContext) {
    super();
    this.context = context;

    // Register the lldb-vscode debug adapter.
    this.pushSubscription(vscode.debug.registerDebugAdapterDescriptorFactory(
        DEBUG_TYPE, new MojoDebugAdapterDescriptorFactory(context)));

    this.pushSubscription(
        vscode.debug.onDidStartDebugSession(async (listener) => {
          if (listener.configuration.type != DEBUG_TYPE)
            return;
          if (!listener.configuration.runInTerminal)
            await vscode.commands.executeCommand(
                "workbench.debug.action.focusRepl");
        }));

    this.pushSubscription(initializeInlineLocalVariablesProvider(context));

    this.pushSubscription(vscode.debug.registerDebugConfigurationProvider(
        DEBUG_TYPE,
        new MojoDebugConfigurationResolver(context),
        ));

    this.pushSubscription(
        vscode.debug.registerDebugConfigurationProvider(
            DEBUG_TYPE,
            new MojoDebugDynamicConfigurationProvider(),
            vscode.DebugConfigurationProviderTriggerKind.Dynamic,
            ),
    );

    this.pushSubscription(activatePickProcessToAttachCommand(this.context));

    this.pushSubscription(vscode.commands.registerCommand(
        "mojo.attachToProcess",
        () => {
          return vscode.debug.startDebugging(undefined, {
            type : "mojo-lldb",
            request : "attach",
            name : "Mojo: Attach to process command",
            pid : "${command:pickProcessToAttach}"
          });
        }))

        // Register the URI-based debug launcher.
        this.pushSubscription(vscode.window.registerUriHandler(
            new UriLaunchServer(context.loggingService)));

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
      let options = config.get<{port : number, secret?: string}>(
          'lldb.rpcServer', folder);
      if (options != null && Object.keys(options).length > 0 &&
          options.port !== undefined) {
        this.updateOrCreateRpcServer(options.port, options.secret);
      }
    }
    if (!config.get<boolean>('lldb.disableDefaultRPCServer',
                             /*workspaceFolder=*/ undefined))
      this.updateOrCreateRpcServer(12346);
  }

  /**
   * Create a debug RPC server with the port and secret.
   */
  private updateOrCreateRpcServer(port: number, secret?: string) {
    const key = `${port}`;
    const existingServer = this.rpcServers.get(key);
    if (existingServer) {
      existingServer.addServerSecret(secret);
    } else {
      let rpcServer =
          new RpcLaunchServer(this.context.loggingService, port, secret || "");
      this.context.loggingService.main.logInfo(
          `Starting RPC server for port '${port}' and secret '${secret}'`);
      this.pushSubscription(rpcServer);
      rpcServer.listen();
      this.rpcServers.set(key, rpcServer);
    }
  }
}
