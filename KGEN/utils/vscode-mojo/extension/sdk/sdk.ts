//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { Logger } from '../logging';
import { MAXSDKConfig } from './sdkConfig';
import { Memoize } from 'typescript-memoize';
import * as util from 'util';
import { MAXSDKKind as MAXSDKKind } from './types';
const execFile = util.promisify(require('child_process').execFile);

/**
 * Class that represents an SDK in the system.
 */
export class MAXSDK {
  public readonly config: MAXSDKConfig;
  public readonly kind: MAXSDKKind;
  private logger: Logger;

  constructor(config: MAXSDKConfig, kind: MAXSDKKind, logger: Logger) {
    this.config = config;
    this.kind = kind;
    this.logger = logger;
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
        this.logger.info('Python scripting support in LLDB found.');
        return true;
      } else {
        this.logger.info(
          `Python scripting support in LLDB not found. The test script returned:\n${
            stdout
          }\n${stderr}`,
        );
      }
    } catch (e) {
      this.logger.error(
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
