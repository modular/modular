//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { checkNsightInstall } from '../utils/checkNsight';
import { DisposableContext } from '../utils/disposableContext';
import { getAllOpenMojoFiles, WorkspaceAwareFile } from '../utils/files';
import { activatePickProcessToAttachCommand } from './attachQuickPick';
import { initializeInlineLocalVariablesProvider } from './inlineVariables';
import { MAXSDK } from '../sdk/sdk';
import { MojoExtension } from '../extension';
import { MAXSDKManager } from '../sdk/sdkManager';
import { quote } from 'shell-quote';
import * as util from 'util';
const execFile = util.promisify(require('child_process').execFile);

/**
 * Stricter version of vscode.DebugConfiguration intended to reduce the chances
 * of typos when handling individual attributes.
 */
export type MojoDebugConfiguration = {
  type?: string;
  name?: string;
  pid?: string | number;
  request?: string;
  modularHomePath?: string;
  modularConfigMojoSection?: string;
  args?: string[];
  program?: string;
  mojoFile?: string;
  env?: string[];
  enableAutoVariableSummaries?: boolean;
  commandEscapePrefix?: string;
  timeout?: number;
  initCommands?: string[];
  customFrameFormat?: string;
  runInTerminal?: boolean;
  buildArgs?: string[];
  enableSyntheticChildDebugging?: boolean;
};

/**
 * Stricter version of vscode.DebugConfiguration intended to reduce the chances
 * of typos when handling individual attributes.
 */
type MojoCudaGdbDebugConfiguration = {
  type?: string;
  description?: string;
  name?: string;
  pid?: string | number;
  modularHomePath?: string;
  args?: string[];
  program?: string;
  mojoFile?: string;
  env?: string[];
  cwd?: string[];
  initCommands?: string[];
  stopOnEntry?: boolean;
  breakOnLaunch?: boolean;
};

/**
 * The "type" for debug configurations.
 */
const DEBUG_TYPE: string = 'mojo-lldb';

/**
 * Some debug configurations come from an RPC call, which have an explicit SDK
 * to use. We should honor it when running the debug session.
 */
async function findSDKForDebugConfiguration(
  config: MojoDebugConfiguration,
  sdkManager: MAXSDKManager,
): Promise<Optional<MAXSDK>> {
  if (config.modularHomePath !== undefined) {
    return sdkManager.createAdHocSDKAndShowError(config.modularHomePath);
  }
  return sdkManager.findSDK(/*hideRepeatedErrors=*/ false);
}
/**
 * This class defines a factory used to find the lldb-vscode binary to use
 * depending on the session configuration.
 */
class MojoDebugAdapterDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  private sdkManager: MAXSDKManager;

  constructor(sdkManager: MAXSDKManager) {
    this.sdkManager = sdkManager;
  }

  async createDebugAdapterDescriptor(
    session: vscode.DebugSession,
    _executable: Optional<vscode.DebugAdapterExecutable>,
  ): Promise<Optional<vscode.DebugAdapterDescriptor>> {
    let sdk = await findSDKForDebugConfiguration(
      session.configuration,
      this.sdkManager,
    );

    // We don't need to show error messages here because
    // `findSDKConfigForDebugSession` does that.
    if (!sdk) {
      this.sdkManager.logger.error(
        "Couldn't find an SDK for the debug session",
      );
      return undefined;
    }
    this.sdkManager.logger.info(
      `Using the SDK ${sdk.config.version.toString()} for the debug session`,
    );
    if (sdk.config.modularHomePath.endsWith('.derived')) {
      // Debug adapters from dev sdks tend to be corrupted because dependencies
      // might need to be rebuilt, so we run a simple verification.
      try {
        await execFile(sdk.config.mojoLLDBVSCodePath, ['--help']);
      } catch (ex: any) {
        const { stderr, stdout } = ex;
        this.sdkManager.logger.main.outputChannel.appendLine(
          '\n\n\n===== LLDB Debug Adapter verification =====',
        );
        this.sdkManager.logger.error(
          'Unable to execute the LLDB Debug Adapter.',
          ex,
        );
        if (stdout) {
          this.sdkManager.logger.info('stdout: ' + stdout);
        }
        if (stderr) {
          this.sdkManager.logger.info('stderr: ' + stderr);
        }
        this.sdkManager.logger.main.outputChannel.show();

        this.sdkManager.showBazelwRunInstallPrompt(
          'The LLDB Debug Adapter seems to be corrupted.',
          sdk.config.modularHomePath,
        );
      }
    }

    return new vscode.DebugAdapterExecutable(sdk.config.mojoLLDBVSCodePath, [
      '--repl-mode',
      'variable',
      '--pre-init-command',
      `?!plugin load '${sdk.config.mojoLLDBPluginPath}'`,
    ]);
  }
}

/**
 * This class defines a factory used to for mojo-cuda-gdb.
 */
class MojoCudaGdbDebugAdapterDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  async createDebugAdapterDescriptor(
    _session: vscode.DebugSession,
    _executable: Optional<vscode.DebugAdapterExecutable>,
  ): Promise<Optional<vscode.DebugAdapterDescriptor>> {
    // We never actually call this, but we need a stub for registration.
    // Instead of making a DebugAdapterDescriptor, we end up tossing the
    // configuration to the Nsight extension by rewriting to their config.
    return undefined;
  }
}

/**
 * This class modifies the debug configuration right before the debug adapter is
 * launched. In other words, this is where we configure lldb-vscode.
 */
class MojoDebugConfigurationResolver
  implements vscode.DebugConfigurationProvider
{
  private sdkManager: MAXSDKManager;

  constructor(sdkManager: MAXSDKManager) {
    this.sdkManager = sdkManager;
  }

  async resolveDebugConfigurationWithSubstitutedVariables?(
    _folder: Optional<vscode.WorkspaceFolder>,
    debugConfiguration: MojoDebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<undefined | vscode.DebugConfiguration> {
    let sdk = await findSDKForDebugConfiguration(
      debugConfiguration,
      this.sdkManager,
    );
    // We don't need to show error messages here because
    // `findSDKConfigForDebugSession` does that.
    if (!sdk) {
      return undefined;
    }

    if (typeof debugConfiguration.pid === 'string') {
      debugConfiguration.pid = parseInt(debugConfiguration.pid);
    }

    if (debugConfiguration.mojoFile) {
      if (
        !debugConfiguration.mojoFile.endsWith('.🔥') &&
        !debugConfiguration.mojoFile.endsWith('.mojo')
      ) {
        const message = `Mojo Debug error: the file '${
          debugConfiguration.mojoFile
        }' doesn't have the .🔥 or .mojo extension.`;
        this.sdkManager.logger.error(message);
        vscode.window.showErrorMessage(message);
        return undefined;
      }
      debugConfiguration.args = [
        'run',
        '--no-optimization',
        '--debug-level',
        'full',
        ...(debugConfiguration.buildArgs || []),
        debugConfiguration.mojoFile,
        ...(debugConfiguration.args || []),
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
        '${function.name-with-args}{${frame.is-artificial} [artificial]}';
    }

    if (debugConfiguration.enableSyntheticChildDebugging === undefined) {
      debugConfiguration.enableSyntheticChildDebugging = true;
    }

    // This setting indicates LLDB to generate a useful summary for each
    // non-primitive type that is displayed right away in the IDE.
    if (debugConfiguration.enableAutoVariableSummaries === undefined) {
      debugConfiguration.enableAutoVariableSummaries = true;
    }

    // This setting indicates LLDB to use the `:` prefix in the Debug Console to
    // disambiguate variable printing from regular LLDB commands.
    if (debugConfiguration.commandEscapePrefix === undefined) {
      debugConfiguration.commandEscapePrefix = ':';
    }

    // This timeout affects targets created with "attachCommands" or
    // "launchCommands".
    if (debugConfiguration.timeout === undefined) {
      debugConfiguration.timeout = initializationTimeoutSec;
    }

    // This setting shortens the length of address strings.
    const initCommands = [
      '?settings set target.show-hex-variable-values-with-leading-zeroes false',
      // FIXME(#23274): remove this when we properly emit the opt flag.
      '?settings set target.process.optimization-warnings false',
      '?mojo statistics telemetry session.start vscode',
    ];

    debugConfiguration.initCommands = [
      ...initCommands,
      ...(debugConfiguration.initCommands || []),
    ];

    // Pull in the additional visualizers within the lldb-visualizers dir.
    if (await sdk.lldbHasPythonScriptingSupport()) {
      let visualizersDir = sdk.config.mojoLLDBVisualizersPath;
      let visualizers = await vscode.workspace.fs.readDirectory(
        vscode.Uri.file(visualizersDir),
      );
      let visualizerCommands = visualizers.map(
        ([name, _type]) => `?command script import ${visualizersDir}/${name}`,
      );
      debugConfiguration.initCommands.push(...visualizerCommands);
    }

    const env = [
      `LLDB_VSCODE_RIT_TIMEOUT_IN_MS=${initializationTimeoutSec * 1000}`, // runInTerminal initialization timeout.
    ];

    env.push(`MODULAR_HOME=${sdk.config.modularHomePath}`);

    debugConfiguration.env = [...env, ...(debugConfiguration.env || [])];
    return debugConfiguration as vscode.DebugConfiguration;
  }

  async resolveDebugConfiguration(
    _folder: Optional<vscode.WorkspaceFolder>,
    debugConfiguration: MojoDebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration> {
    // The `Debug: Start Debugging` command (aka F5 or the `Run and Debug`
    // button if no launch.json files are present), invoke this method with a
    // totally empty debugConfiguration, so we have to fill it in.
    if (!debugConfiguration.request) {
      debugConfiguration.type = DEBUG_TYPE;
      debugConfiguration.request = 'launch';
      // This will get replaced with the currently active document.
      debugConfiguration.mojoFile = '${file}';
    }

    return debugConfiguration as vscode.DebugConfiguration;
  }
}

/**
 * This class modifies the debug configuration right before the debug adapter is
 * launched. This is where we mutate the mojo-cuda-gdb config into normal
 * cuda-gdb config.
 */
class MojoCudaGdbDebugConfigurationResolver
  implements vscode.DebugConfigurationProvider
{
  private sdkManager: MAXSDKManager;

  constructor(sdkManager: MAXSDKManager) {
    this.sdkManager = sdkManager;
  }

  async resolveDebugConfigurationWithSubstitutedVariables?(
    _folder: Optional<vscode.WorkspaceFolder>,
    debugConfigIn: MojoCudaGdbDebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<undefined | vscode.DebugConfiguration> {
    const maybeErrorMessage = await checkNsightInstall(this.sdkManager.logger);
    if (maybeErrorMessage) {
      return undefined;
    }

    // relax the debugConfig.args type
    const debugConfig = debugConfigIn as vscode.DebugConfiguration;
    let args = debugConfigIn.args || [];

    let sdk = await findSDKForDebugConfiguration(
      debugConfigIn as vscode.DebugConfiguration,
      this.sdkManager,
    );
    // We don't need to show error messages here because
    // `findSDKConfigForDebugSession` does that.
    if (!sdk) {
      return undefined;
    }
    // If we have a mojoFile config, translate it to program plus args.
    if (debugConfigIn.mojoFile) {
      debugConfig.program = sdk.config.mojoDriverPath;
      args = [
        'run',
        '--no-optimization',
        '--debug-level',
        'full',
        debugConfig.mojoFile,
        ...args,
      ];
    }

    // Transform debugConfig into normal cuda-gdb config.
    debugConfig.type = 'cuda-gdb';
    // cuda-gdb takes args as a single string, while we take them as an array.
    // Actually, cuda-gdb can take an array, which it then joins into a single
    // string separated by ";" characters. So it takes the list of program
    // arguments to the debuggee as a single string.
    debugConfig.args = quote(args || []);
    // cuda-gdb takes environment as a list of objects like:
    // [{"name": "HOME", "value": "/home/ubuntu"}]
    let env = [];
    env.push(`MODULAR_HOME=${sdk.config.modularHomePath}`);
    env = [...env, ...(debugConfigIn.env || [])];
    debugConfig.environment = env.map((envStr: String) => {
      const split = envStr.split('=');
      return { name: split[0], value: split.slice(1).join('=') };
    });
    // Minor name changes between cuda-gdb and mojo-cuda-gdb...
    debugConfig.stopAtEntry = debugConfigIn.stopOnEntry;
    debugConfig.processId = debugConfigIn.pid;
    return debugConfig;
  }
}

/**
 * Provides debug configurations dynamically depending on the currently open
 * workspaces and files.
 */
class MojoDebugDynamicConfigurationProvider
  implements vscode.DebugConfigurationProvider
{
  async provideDebugConfigurations(
    _folder: Optional<vscode.WorkspaceFolder>,
    _token?: Optional<vscode.CancellationToken>,
  ): Promise<Optional<vscode.DebugConfiguration[]>> {
    const [activeFile, otherOpenFiles] = getAllOpenMojoFiles();
    return [activeFile, ...otherOpenFiles]
      .filter((file): file is WorkspaceAwareFile => !!file)
      .map((file) => {
        return {
          type: DEBUG_TYPE,
          request: 'launch',
          name: `Mojo: Debug ${file.baseName} ⸱ ${file.relativePath}`,
          mojoFile: file.uri.fsPath,
          args: [],
          env: [],
          cwd: file.workspaceFolder?.uri.fsPath,
          runInTerminal: false,
        };
      });
  }
}

/**
 * Class used to register and manage all the necessary constructs to support
 * mojo debugging.
 */
export class MojoDebugManager extends DisposableContext {
  private sdkManager: MAXSDKManager;

  constructor(extension: MojoExtension, sdkManager: MAXSDKManager) {
    super();
    this.sdkManager = sdkManager;

    // Register the lldb-vscode debug adapter.
    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        DEBUG_TYPE,
        new MojoDebugAdapterDescriptorFactory(this.sdkManager),
      ),
    );

    this.pushSubscription(
      vscode.debug.onDidStartDebugSession(async (listener) => {
        if (listener.configuration.type != DEBUG_TYPE) {
          return;
        }

        if (!listener.configuration.runInTerminal) {
          await vscode.commands.executeCommand('workbench.view.debug');
          await vscode.commands.executeCommand(
            'workbench.debug.action.focusRepl',
          );
        }
      }),
    );

    this.pushSubscription(initializeInlineLocalVariablesProvider(extension));

    this.pushSubscription(
      vscode.debug.registerDebugConfigurationProvider(
        DEBUG_TYPE,
        new MojoDebugConfigurationResolver(sdkManager),
      ),
    );

    this.pushSubscription(
      vscode.debug.registerDebugConfigurationProvider(
        DEBUG_TYPE,
        new MojoDebugDynamicConfigurationProvider(),
        vscode.DebugConfigurationProviderTriggerKind.Dynamic,
      ),
    );

    this.pushSubscription(
      activatePickProcessToAttachCommand(extension.extensionContext),
    );

    this.pushSubscription(
      vscode.commands.registerCommand('mojo.debug.attach-to-process', () => {
        return vscode.debug.startDebugging(undefined, {
          type: 'mojo-lldb',
          request: 'attach',
          name: 'Mojo: Attach to process command',
          pid: '${command:pickProcessToAttach}',
        });
      }),
    );

    // Add subscriptions for mojo-cuda-gdb.  Need to register
    // DAPDescriptorFactory, but all of the real work is handled by
    // registerDebugConfigurationProvider, which translates the config to
    // Nsight's cuda-gdb format, ultimately launching its debugger instead.
    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        'mojo-cuda-gdb',
        new MojoCudaGdbDebugAdapterDescriptorFactory(),
      ),
    );

    this.pushSubscription(
      vscode.debug.registerDebugConfigurationProvider(
        'mojo-cuda-gdb',
        new MojoCudaGdbDebugConfigurationResolver(sdkManager),
      ),
    );
  }
}
