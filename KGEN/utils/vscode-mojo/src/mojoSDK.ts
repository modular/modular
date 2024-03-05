//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as child_process from 'child_process';
import * as ini from 'ini';
import * as path from 'path';
import * as util from 'util';
import * as vscode from 'vscode';

const execFile = util.promisify(require('child_process').execFile);

import {LoggingService} from './logging';
import * as config from './utils/config';
import {substituteVariables} from './utils/vscodeVariables';
import {isNightlyExtension} from './utils/buildInfo';

/**
 * This class represents a subset of the Modular config object used by extension
 * for interacting with mojo.
 */
export class MOJOSDKConfig {
  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  constructor(loggingService: LoggingService) {
    this.loggingService = loggingService;
  }

  /**
   * The MODULAR_HOME path containing the SDK.
   */
  modularHomePath: string = "";

  /**
   * The path to the mojo driver within the SDK installation.
   */
  mojoDriverPath: string = "";

  /**
   * The path to the LLDB vscode debug adapter.
   */
  mojoLLDBVSCodePath: string = "";

  /**
   * The path to the LLDB visualizers.
   */
  mojoLLDBVisualizersPath: string = "";

  /**
   * The path the mojo language server within the SDK installation.
   */
  mojoLanguageServerPath: string = "";

  /**
   * The path to the mojo LLDB plugin.
   */
  mojoLLDBPluginPath: string = "";

  /**
   * The path to the LLDB binary.
   */
  lldbPath: string = "";

  private lldbHasPythonScriptingSupportResult?: Promise<boolean>;

  /**
   * @returns true if and only if the LLDB binary in this SDK has a working
   *     python scripting feature.
   */
  public lldbHasPythonScriptingSupport(): Promise<boolean> {
    // We cache this check because it's not a no-op.
    if (this.lldbHasPythonScriptingSupportResult == undefined)
      this.lldbHasPythonScriptingSupportResult =
          this.doLLDBHasPythonScriptingSupport();
    return this.lldbHasPythonScriptingSupportResult;
  }

  /**
   * Actually determine whether python scripting is functional in LLDB. As there
   * are many reasons why python scripting would fail (e.g. disabled in CMake,
   * wrong SDK installation, etc.), it's more effective to just execute a
   * minimal script to confirm it's operative.
   */
  private async doLLDBHasPythonScriptingSupport(): Promise<boolean> {
    try {
      let {stdout, stderr} =
          await execFile(this.lldbPath, [ "-b", "-o", "script print(100+1)" ]);
      stdout = (stdout || "") as string;
      stderr = (stderr || "") as string;

      if (stdout.indexOf("101") != -1) {
        this.loggingService.logInfo("Python scripting support in LLDB found.");
        return true;
      } else {
        this.loggingService.logInfo(
            `Python scripting support in LLDB not found. The test script returned:\n${
                stdout}\n${stderr}`);
      }
    } catch (e) {
      this.loggingService.logError(
          "Python scripting support in LLDB not found. The test script failed with",
          e);
    }
    return false;
  }
}

/**
 *  This class manages interacting with and checking the status of the Mojo SDK.
 */
export class MOJOSDK {
  /**
   * Cache for the `resolveConfig` method.
   */
  private resolveConfigCache: Map<string, MOJOSDKConfig> = new Map();

  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  /**
   * The current extension context.
   */
  private readonly context: vscode.ExtensionContext;

  constructor(loggingService: LoggingService,
              context: vscode.ExtensionContext) {
    this.loggingService = loggingService;
    this.context = context;
  }

  /**
   * Return if the given modular home path refers to a dev-build of the SDK.
   */
  private static isDevBuild(modularHomePath: string) {
    return modularHomePath.endsWith('.derived');
  }

  /**
   * Return the configuration key for the SDK within the modular.cfg file.
   */
  private static getConfigKey(modularHomePath: string, isNightly: boolean,
                              possibleKeys: string[]): string|undefined {
    // Bail early if we don't have any keys.
    if (possibleKeys.length === 0)
      return undefined;

    // If this is a dev-build, there'll only be one key so just grab
    // it.
    if (MOJOSDK.isDevBuild(modularHomePath))
      return possibleKeys[0];

    // Filter the keys to only those that match the current extension.
    possibleKeys =
        possibleKeys.filter(key => isNightly == key.endsWith("-nightly"));
    if (possibleKeys.length === 0)
      return undefined;

    // Prefer the 'max' key if it exists.
    const maxKey = possibleKeys.find(key => key.includes("max"));
    if (maxKey)
      return maxKey;

    // Otherwise, just grab the first key.
    return possibleKeys[0];
  }

  /**
   * Emit a warning to the user if the current SDK is out of date.
   */
  private async warnIfSDKOutOfDate(modularHomePath: string, mojoPath: string) {
    // If this is a dev-build, there's no version to check.
    if (MOJOSDK.isDevBuild(modularHomePath))
      return;

    // Otherwise, invoke `mojo` to grab the current version.
    try {
      let rawStdout = child_process.execFileSync(mojoPath, [ "--version" ], {
        env : {...process.env, "MODULAR_HOME" : modularHomePath},
      });
      let stdout = rawStdout.toString();

      // Grab the version string from the output.
      const match = stdout.match(/mojo\s+\b([0-9]+)\.([0-9]+)\.([0-9]+)\b.*/);
      if (!match) {
        this.loggingService.logError(
            "`mojo` returned an unexpected version string: " + stdout);
        return;
      }

      // Grab the current extension version.
      const extensionVersion =
          this.context.extension.packageJSON.version as string;
      const extensionVersionMatch =
          extensionVersion.match(/([0-9]+)\.([0-9]+)\.([0-9]+)/);
      if (!extensionVersionMatch) {
        this.loggingService.logError("Unable to compute extension version: " +
                                     extensionVersion);
        return;
      }

      // Compare the two versions. We don't warn if the extension is older,
      // just if the SDK is older.
      if (/*major*/ +match[1] < +extensionVersionMatch[1] ||
          /*minor*/ +match[2] < +extensionVersionMatch[2] ||
          /*patch*/ +match[3] < +extensionVersionMatch[3]) {
        vscode.window.showWarningMessage(
            "The current Mojo SDK version is incompatible with this " +
            "version of the Mojo extension. Please update your SDK " +
            "to ensure the extension behaves correctly.");
      }
    } catch (e) {
      this.loggingService.logError(
          "Unable to invoke `mojo` to check the SDK version, failed with: ", e);
    }
  }

  /**
   * Resolve the Modular config for the given context.
   *
   * - If `context` contains `sdkPath`, then the resolver will use it as the SDK
   * path.
   * - Otherwise, the resolver will look for the SDK path based on the
   * `mojo.modularHomePath` setting. This doesn't have a consistent behavior,
   * though:
   *   - If `context.workspaceFolder` is provided, then VSCode will search for
   * the setting at the workspace-level, and then at the user-level as fallback.
   *   - If `context.workspaceFolder` is not provided and there's only one
   * workspace mounted, then VSCode will search for the setting in that
   * workspace, and then at user-level as fallback.
   *   - If `context.workspaceFolder` is not provided and there's 0 or more than
   * one workspace mounted, then VSCode will search for the setting only at the
   * user-level.
   * - If the config is not yet found, then the resolver will try to use the
   * MODULAR_HOME environment variable.
   *
   * @param promptSDKInstall Whether to prompt the user to install the SDK if it
   *     is missing.
   */
  public async resolveConfig(context: {
    workspaceFolder?: vscode.WorkspaceFolder,
    sdkPath?: string
  }): Promise<MOJOSDKConfig|undefined> {
    const key = JSON.stringify({
      workspaceFolder : context.workspaceFolder?.uri.fsPath,
      sdkPath : context.sdkPath
    });
    let mojoConfig = this.resolveConfigCache.get(key);
    if (mojoConfig)
      return mojoConfig;

    let modularPath: string|undefined =
        context.sdkPath ||
        await this.tryGetModularHomePathFromConfig(context.workspaceFolder);

    // Otherwise, check to see if the environment variable is set.
    if (modularPath) {
      this.loggingService.logInfo("MODULAR_HOME found in VS Code settings.");
    } else if (process.env.MODULAR_HOME) {
      modularPath = process.env.MODULAR_HOME;
      this.loggingService.logInfo(
          "MODULAR_HOME found as an environment variable.");

      // If we still don't have a path, prompt the user to install the SDK.
    } else {
      this.loggingService.logInfo("MODULAR_HOME not found.");
      this.promptInstallSDK();
      return undefined;
    }

    this.loggingService.logInfo(`MODULAR_HOME is ${modularPath}.`);

    // Read in the config file.
    const modularCfg = path.join(modularPath, "modular.cfg");
    let configPath = vscode.Uri.file(modularCfg);

    try {
      let configPathStat = await vscode.workspace.fs.stat(configPath);
      if (!(configPathStat.type & vscode.FileType.File)) {
        this.showSDKErrorMessage(
            `The modular config file '${modularCfg}' is not a file.`);
        this.promptInstallSDK();
        return undefined;
      }
    } catch (e) {
      this.showSDKErrorMessage(
          `The modular config file '${
              modularCfg}' does not exist or VS Code does not have permissions to access it.`,
          e);
      this.promptInstallSDK();
      return undefined;
    }
    let modularConfig = ini.parse(new TextDecoder().decode(
        await vscode.workspace.fs.readFile(configPath)));
    this.loggingService.logInfo("modular.cfg file with contents",
                                modularConfig);

    // Find the appropriate mojo configuration key in the config file.
    let mojoKeys: string[] =
        Object.keys(modularConfig).filter(key => key.startsWith("mojo"));
    let configKey = MOJOSDK.getConfigKey(
        modularPath, isNightlyExtension(this.context), mojoKeys);
    if (!configKey) {
      this.promptInstallSDK();
      return undefined;
    }
    let modularMojoConfig = modularConfig[configKey];

    // Extract out the pieces of the config that we care about.
    mojoConfig = new MOJOSDKConfig(this.loggingService);
    mojoConfig.modularHomePath = modularPath;
    mojoConfig.mojoLLDBVSCodePath = modularMojoConfig.lldb_vscode_path;
    mojoConfig.mojoLLDBVisualizersPath =
        modularMojoConfig.lldb_visualizers_path;
    mojoConfig.mojoDriverPath = modularMojoConfig.driver_path;
    mojoConfig.mojoLanguageServerPath = modularMojoConfig.lsp_server_path;
    mojoConfig.mojoLLDBPluginPath = modularMojoConfig.lldb_plugin_path;
    mojoConfig.lldbPath = modularMojoConfig.lldb_path;

    // Now that we have a resolved SDK, warn if it's out of date.
    await this.warnIfSDKOutOfDate(modularPath, mojoConfig.mojoDriverPath);

    // Cache the config.
    this.resolveConfigCache.set(key, mojoConfig);
    return mojoConfig;
  }

  /**
   * Prompt to the user that the SDK is missing, and provide a link to the
   * installation instructions.
   */
  private async promptInstallSDK() {
    this.loggingService.logInfo("Prompting Install SDK.")
    let value = await vscode.window.showInformationMessage(
        ("The Mojo🔥 development environment was not found. If the Mojo " +
         "SDK is installed, please set the MODULAR_HOME environment variable to the " +
         "appropriate path, or set the `mojo.modularHomePath` configuration. If you do " +
         "not have it installed, would you like to install it?"),
        "Install", "Open setting");
    if (value === "Install") {
      // TODO: This should resolve to the actual mojo download link when
      // the user console is in place.
      vscode.env.openExternal(vscode.Uri.parse("https://www.modular.com/mojo"));
    } else if (value === "Open setting") {
      vscode.commands.executeCommand(
          'workbench.action.openWorkspaceSettings',
          {openToSide : false, query : `mojo.modularHomePath`});
    }
  }

  /**
   * Attempt to retrieve the modular home path from the config. This will also
   * perform the substitution of some common VSCode variables.
   *
   * If the setting does not exist or the resolved path is not a directory,
   * return undefined.
   */
  private async tryGetModularHomePathFromConfig(workspaceFolder:
                                                    vscode.WorkspaceFolder|
                                                undefined):
      Promise<string|undefined> {
    let modularPath = config.get<string>('modularHomePath', workspaceFolder);
    if (!modularPath)
      return undefined;
    const substituted = substituteVariables(modularPath, workspaceFolder);

    const showError = (reason: string) => {
      let message = `The mojo.modularHomePath setting '${modularPath}'`;
      if (substituted !== modularPath)
        message += `, which resolves to '${substituted}',`;
      message += " " + reason + ".";
      this.showSDKErrorMessage(message);
      return undefined;
    };

    if (substituted.length == 0) {
      return showError("is empty");
    }

    try {
      let configPathStat =
          await vscode.workspace.fs.stat(vscode.Uri.file(substituted));
      if (configPathStat.type & vscode.FileType.Directory)
        return substituted;
      return showError("is not a directory");
    } catch (err) {
      return showError("does not exist");
    }
  }

  /**
   * Show an error message as a VSCode notification and log it to the output
   * channel as well.
   */
  private showSDKErrorMessage(message: string, error?: unknown): void {
    message = "Mojo SDK initialization error: " + message;
    this.loggingService.logError(message, error);
    vscode.window.showErrorMessage(message);
  }
}
