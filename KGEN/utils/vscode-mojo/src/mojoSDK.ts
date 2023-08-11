//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as ini from 'ini';
import * as path from 'path';
import * as vscode from 'vscode';

import * as config from './config';
import {LoggingService} from './logging';

/**
 * This class represents a subset of the Modular config object used by extension
 * for interacting with mojo.
 */
export class MOJOSDKConfig {
  /**
   * The MODULAR_HOME path containing the SDK.
   */
  modularHomePath: string = "";

  /**
   * The path to the mojo driver within the SDK installation.
   */
  mojoDriverPath: string = "";

  /**
   * The path the mojo language server within the SDK installation.
   */
  mojoLanguageServerPath: string = "";
}

/**
 *  This class manages interacting with and checking the status of the Mojo SDK.
 */
export class MOJOSDK {
  /**
   * The resolved Modular config for a set of workspaces.
   */
  workspaceConfigs: Map<string, MOJOSDKConfig> = new Map();

  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  constructor(loggingService: LoggingService) {
    this.loggingService = loggingService;
  }

  /**
   * Resolve the Modular config for the given workspace directory.
   *
   * @param workspaceFolder The current workspace folder, or undefined.
   * @param promptSDKInstall Whether to prompt the user to install the SDK
   *                            if it is missing.
   */
  public async resolveConfig(workspaceFolder: vscode.WorkspaceFolder|
                             undefined): Promise<MOJOSDKConfig|undefined> {
    let workspaceFolderStr =
        workspaceFolder ? workspaceFolder.uri.toString() : "";
    let mojoConfig = this.workspaceConfigs.get(workspaceFolderStr);
    if (mojoConfig)
      return mojoConfig;

    // Check to see if a path was specified in the config.
    let modularPath = config.get<string>('modularHomePath', workspaceFolder);

    // Otherwise, check to see if the environment variable is set.
    if (!modularPath) {
      modularPath = process.env.MODULAR_HOME;
    } else {
      this.loggingService.logInfo("MODULAR_HOME found in VS Code settings.");
      let workspaceRoot = workspaceFolder ? workspaceFolder.uri.fsPath : "";
      modularPath = modularPath.replace("${workspaceRoot}", workspaceRoot);
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
    let configPath = vscode.Uri.from({scheme : 'file', path : modularCfg});
    let configPathStat = await vscode.workspace.fs.stat(configPath);
    if (!(configPathStat.type & vscode.FileType.File)) {
      this.loggingService.logInfo(
          `Missing or invalid modular.cfg file: '${modularCfg}'.`);
      this.promptInstallSDK();
      return undefined;
    }
    let modularConfig = ini.parse(new TextDecoder().decode(
        await vscode.workspace.fs.readFile(configPath)));

    this.loggingService.logInfo("modular.cfg file with contents",
                                modularConfig);

    // Extract out the pieces of the config that we care about.
    mojoConfig = new MOJOSDKConfig();
    mojoConfig.modularHomePath = modularPath;
    mojoConfig.mojoDriverPath = modularConfig.mojo.driver_path;
    mojoConfig.mojoLanguageServerPath = modularConfig.mojo.lsp_server_path;

    // Cache the config for the workspace.
    this.workspaceConfigs.set(workspaceFolderStr, mojoConfig);
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
        "install", "open setting");
    if (value === "install") {
      // TODO: This should resolve to the actual mojo download link when
      // the user console is in place.
      vscode.env.openExternal(vscode.Uri.parse("https://www.modular.com/mojo"));
    } else if (value === "open setting") {
      vscode.commands.executeCommand(
          'workbench.action.openWorkspaceSettings',
          {openToSide : false, query : `mojo.modularHomePath`});
    }
  }
}
