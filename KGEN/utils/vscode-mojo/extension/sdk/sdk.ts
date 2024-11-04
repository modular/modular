//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { Logger } from '../logging';
import { MojoSDKConfig } from './sdkConfig';
import { Memoize } from 'typescript-memoize';
import * as util from 'util';
import { MojoSDKKind } from './types';
const execFile = util.promisify(require('child_process').execFile);

/**
 * Class that represents an SDK in the system.
 */
export class MojoSDK {
  public readonly config: MojoSDKConfig;
  public readonly kind: MojoSDKKind;
  private logger: Logger;

  constructor(config: MojoSDKConfig, kind: MojoSDKKind, logger: Logger) {
    this.config = config;
    this.kind = kind;
    this.logger = logger;
  }

  /**
   * Return the configuration key for the SDK within the modular.cfg file.
   */
  static getConfigKey(
    modularHomePath: string,
    isNightly: boolean,
    possibleKeys: string[],
  ): Optional<string> {
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
      (key) => isNightly === key.endsWith('-nightly'),
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
        this.logger.main.logInfo('Python scripting support in LLDB found.');
        return true;
      } else {
        this.logger.main.logInfo(
          `Python scripting support in LLDB not found. The test script returned:\n${
            stdout
          }\n${stderr}`,
        );
      }
    } catch (e) {
      this.logger.main.logError(
        'Python scripting support in LLDB not found. The test script failed with',
        e,
      );
    }
    return false;
  }

  /**
   * Returns a process environment to be used when executing SDK
   * binaries.
   */
  public getProcessEnv(withTelemetry: boolean = true): NodeJS.ProcessEnv {
    let env = { ...process.env };

    // If we had modular home provided somewhere, make sure that
    // gets propagated.
    if (this.config.modularHomePath) {
      env.MODULAR_HOME = this.config.modularHomePath;
    }
    if (!withTelemetry) {
      env.MODULAR_TELEMETRY_ENABLED = 'false';
    }
    return env;
  }
}
