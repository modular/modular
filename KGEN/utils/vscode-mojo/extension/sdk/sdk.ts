//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import { LoggingService } from '../logging';
import { MojoSDKConfig } from './sdkConfig';
import { Memoize } from 'typescript-memoize';
import * as util from 'util';
const execFile = util.promisify(require('child_process').execFile);

/**
 * Class that represents an SDK in the system.
 */
export class MojoSDK {
  public readonly config: MojoSDKConfig;
  private loggingService: LoggingService;
  private context: vscode.ExtensionContext;

  constructor(
    config: MojoSDKConfig,
    loggingService: LoggingService,
    context: vscode.ExtensionContext
  ) {
    this.config = config;
    this.loggingService = loggingService;
    this.context = context;
  }

  /**
   * Return the configuration key for the SDK within the modular.cfg file.
   */
  static getConfigKey(
    modularHomePath: string,
    isNightly: boolean,
    possibleKeys: string[]
  ): string | undefined {
    // Bail early if we don't have any keys.
    if (possibleKeys.length === 0) {
      return undefined;
    }

    // If this is a dev-build path, there'll only be one key so just grab
    // it.
    if (modularHomePath.endsWith('.derived')) {
      return possibleKeys[0];
    }

    // Filter the keys to only those that match the current extension.
    possibleKeys = possibleKeys.filter(
      (key) => isNightly == key.endsWith('-nightly')
    );

    if (possibleKeys.length === 0) {
      return undefined;
    }

    // Prefer the 'max' key if it exists.
    const maxKey = possibleKeys.find((key) => key.includes('max'));

    if (maxKey) {
      return maxKey;
    }

    // Otherwise, just grab the first key.
    return possibleKeys[0];
  }

  /**
   * Emit a warning to the user if the current SDK is out of date.
   */
  public async warnIfSDKOutOfDate() {
    // If this is a dev-build, there's no version to check.
    if (this.config.version.isDev()) {
      return;
    }

    // Grab the current extension version.
    const extensionVersion = this.context.extension.packageJSON
      .version as string;
    const extensionVersionMatch = extensionVersion.match(
      /([0-9]+)\.([0-9]+)\.([0-9]+)/
    );
    if (!extensionVersionMatch) {
      this.loggingService.main.logError(
        'Unable to compute extension version: ' + extensionVersion
      );
      return;
    }

    // Compare the two versions. We don't warn if the extension is older,
    // just if the SDK is older.
    if (
      this.config.version.major < +extensionVersionMatch[1] ||
      this.config.version.minor < +extensionVersionMatch[2] ||
      this.config.version.patch < +extensionVersionMatch[3]
    ) {
      vscode.window.showWarningMessage(
        'The current Mojo SDK version is incompatible with this ' +
          'version of the Mojo extension. Please update your SDK ' +
          'to ensure the extension behaves correctly.'
      );
    }
  }

  /**
   * Determine whether python scripting is functional in LLDB. As there
   * are many reasons why python scripting would fail (e.g. disabled in the build system,
   * wrong SDK installation, etc.), it's more effective to just execute a
   * minimal script to confirm it's operative.
   *
   * @returns true if and only if the LLDB binary in this SDK has a working
   *     python scripting feature.
   */
  @Memoize()
  public async lldbHasPythonScriptingSupport(): Promise<boolean> {
    try {
      let { stdout, stderr } = await execFile(this.config.lldbPath, [
        '-b',
        '-o',
        'script print(100+1)',
      ]);
      stdout = (stdout || '') as string;
      stderr = (stderr || '') as string;

      if (stdout.indexOf('101') != -1) {
        this.loggingService.main.logInfo(
          'Python scripting support in LLDB found.'
        );
        return true;
      } else {
        this.loggingService.main.logInfo(
          `Python scripting support in LLDB not found. The test script returned:\n${
            stdout
          }\n${stderr}`
        );
      }
    } catch (e) {
      this.loggingService.main.logError(
        'Python scripting support in LLDB not found. The test script failed with',
        e
      );
    }
    return false;
  }
}
