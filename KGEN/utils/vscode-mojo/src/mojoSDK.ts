//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as ini from 'ini';
import * as path from 'path';
import * as util from 'util';
import * as vscode from 'vscode';

const execFile = util.promisify(require('child_process').execFile);

import {LoggingService} from './logging';
import * as config from './utils/config';
import {substituteVariables} from './utils/vscodeVariables';

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
   * The resolved Modular config for a set of workspaces.
   */
  private workspaceConfigs: Map<string, MOJOSDKConfig> = new Map();

  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  constructor(loggingService: LoggingService) {
    this.loggingService = loggingService;
  }

  /**
   * Resolve the Modular config for the given context.
   *
   * - If `context` is a string, then the resolver will use it as the SDK path.
   * - If `context` is a WorkspaceFolder, then the resolver will look for the
   * mojo.modularHomePath the WorkspaceFolder's settings and use it as the SDK
   * path. In case of failures, it'll resort to the global settings for the IDE.
   * - If `context` is undefined, then the resolver will look at the global
   * settings of the IDE.
   * - If the SDK was not found in the previous checks, then the resolver will
   * look for MODULAR_HOME in the environment of the IDE process.
   *
   * @param context The current workspace folder if its type is
   *     vscode.WorkspaceFolder, or the Mojo SDK path if its type is string, or
   *     undefined.
   * @param promptSDKInstall Whether to prompt the user to install the SDK if it
   *     is missing.
   */
  public async resolveConfig(context: vscode.WorkspaceFolder|string|
                             undefined): Promise<MOJOSDKConfig|undefined> {
    let key = "";
    if (typeof context === "string") {
      key = context;
    } else if (context) {
      key = context.uri.fsPath;
    }

    let mojoConfig = this.workspaceConfigs.get(key);
    if (mojoConfig)
      return mojoConfig;

    let modularPath: string|undefined;

    // Check to see if a path was specified explicitly.
    if (typeof (context) == "string") {
      modularPath = context;

      // Check to see if a path was specified in the config.
    } else {
      modularPath = await this.tryGetModularHomePathFromConfig(context);
    }

    // Otherwise, check to see if the environment variable is set.
    if (!modularPath) {
      modularPath = process.env.MODULAR_HOME;
    } else {
      this.loggingService.logInfo("MODULAR_HOME found in VS Code settings.");
    }

    // If we still don't have a path, prompt the user to install the SDK.
    if (!modularPath) {
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

    // Extract out the pieces of the config that we care about.
    mojoConfig = new MOJOSDKConfig(this.loggingService);
    mojoConfig.modularHomePath = modularPath;
    mojoConfig.mojoLLDBVSCodePath = modularConfig.mojo.lldb_vscode_path;
    mojoConfig.mojoLLDBVisualizersPath =
        modularConfig.mojo.lldb_visualizers_path;
    mojoConfig.mojoDriverPath = modularConfig.mojo.driver_path;
    mojoConfig.mojoLanguageServerPath = modularConfig.mojo.lsp_server_path;
    mojoConfig.mojoLLDBPluginPath = modularConfig.mojo.lldb_plugin_path;
    mojoConfig.lldbPath = modularConfig.mojo.lldb_path;

    // Cache the config for the workspace.
    this.workspaceConfigs.set(key, mojoConfig);
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
