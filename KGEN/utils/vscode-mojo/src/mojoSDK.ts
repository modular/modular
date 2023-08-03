//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

/**
 *  This class manages interacting with and checking the status of the Mojo SDK.
 */
export class MOJOSDK {
  /**
   * The resolved Mojo SDK path, or empty if the SDK isn't installed.
   */
  private mojoSDKPath: string|undefined;

  /**
   * Construct a new MOJOSDK object.
   */
  public constructor() { this.resolveMojoSDKPath(); }

  /**
   * Resolve a file within the Mojo SDK. Returns an empty string if the SDK is
   * not installed or in the case of error.
   *
   * @param path The path to resolve within the SDK.
   * @param promptSDKInstall Whether to prompt the user to install the SDK
   *                            if it is missing.
   */
  public async resolvePath(path: string,
                           promptSDKInstall: boolean): Promise<string> {
    // Try to resolve the SDK path again if we didn't find it before.
    if (!this.mojoSDKPath) {
      await this.resolveMojoSDKPath(promptSDKInstall);
      if (!this.mojoSDKPath)
        return "";
    }

    // Resolve the path within the SDK.
    let foundUris = await vscode.workspace.findFiles(
        new vscode.RelativePattern(this.mojoSDKPath, '**/' + path), null, 1);
    if (foundUris.length === 0)
      return "";
    return foundUris[0].fsPath;
  }

  /**
   * Resolve the path of the Mojo SDK, prompting the user with how to install if
   * the SDK is missing.
   *
   * @param promptSDKInstall Whether to prompt the user to install the SDK
   *                            if it is missing.
   */
  private async resolveMojoSDKPath(promptSDKInstall: boolean = true) {
    // Check for a development version of the SDK.
    if (process.env.MODULAR_PATH) {
      this.mojoSDKPath = process.env.MODULAR_PATH;
      return;
    }

    // Check to see if Modular is installed at all.
    if (!process.env.MODULAR_HOME) {
      if (promptSDKInstall)
        this.promptInstallSDK();
      return;
    }
    const sdkPath = path.join(process.env.MODULAR_HOME, "pkg", "mojo");
    fs.stat(sdkPath, (err, stats) => {
      if (err || !stats.isDirectory()) {
        if (promptSDKInstall)
          this.promptInstallSDK();
        return;
      }
      this.mojoSDKPath = sdkPath;
    });
  }

  /**
   * Prompt to the user that the SDK is missing, and provide a link to the
   * installation instructions.
   */
  private async promptInstallSDK() {
    let value = await vscode.window.showInformationMessage(
        "The Mojo🔥 development environment was not found. Would you like to install it?",
        "install");
    if (value === "install") {
      // TODO: This should resolve to the actual mojo download link when
      // the user console is in place.
      vscode.env.openExternal(vscode.Uri.parse("https://www.modular.com/mojo"));
    }
  }
}
