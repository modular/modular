//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as util from 'util';

const execFile = util.promisify(require('child_process').execFile);

import { LoggingService } from '../logging';
import { MojoSDKVersion } from './sdkVersion';


/**
 * This class represents a subset of the Modular config object used by extension
 * for interacting with mojo.
 */
export class MojoSDKConfig {
  /**
   * Create a new MojoSDKConfig object from the given configuration.
   */
  static async create(
    loggingService: LoggingService,
    modularPath: string,
    configSection: string,
    rawConfig: { [key: string]: any }
  ): Promise<MojoSDKConfig | undefined> {
    let version = await MojoSDKConfig.parseVersionFromDriver(
      loggingService,
      rawConfig.driver_path,
      configSection
    );

    if (!version) {
      return undefined;
    }
    return new MojoSDKConfig(loggingService, version, modularPath, rawConfig);
  }

  /**
   * Returns a process environment to be used when executing SDK
   * binaries.
   */
  public getProcessEnv(): NodeJS.ProcessEnv {
    let env = { ...process.env };

    // If we had modular home provided somewhere, make sure that
    // gets propagated.
    if (this.modularHomePath) {
      env.MODULAR_HOME = this.modularHomePath;
    }
    return env;
  }

  /**
   * @returns true if and only if the LLDB binary in this SDK has a working
   *     python scripting feature.
   */
  public lldbHasPythonScriptingSupport(): Promise<boolean> {
    // We cache this check because it's not a no-op.
    if (this.lldbHasPythonScriptingSupportResult == undefined) {
      this.lldbHasPythonScriptingSupportResult =
        this.doLLDBHasPythonScriptingSupport();
    }
    return this.lldbHasPythonScriptingSupportResult;
  }

  /**
   * Parse a version number from the given mojo driver.
   */
  private static async parseVersionFromDriver(
    loggingService: LoggingService,
    driverPath: string,
    configSection: string
  ): Promise<MojoSDKVersion | undefined> {
    try {
      let { stdout, stderr } = await execFile(driverPath, ['--version'], {
        env: { ...process.env },
        encoding: 'utf-8',
      });

      if (stderr) {
        return undefined;
      }

      let match = stdout
        .toString()
        .match(/mojo\s+([0-9]+)\.([0-9]+)\.([0-9]+)/);

      if (!match) {
        return undefined;
      }

      // Build the title of the version based on the config key.
      let title = 'Mojo';

      if (configSection.includes('max')) {
        title += ' Max';
      }

      if (configSection.includes('nightly')) {
        title += ' (nightly)';
      }

      return new MojoSDKVersion(
        title,
        +match[1],
        +match[2],
        +match[3],
        driverPath
      );
    } catch (e) {
      loggingService.main.logError(
        'Unable to parse version from `mojo` driver: ',
        e
      );
      return undefined;
    }
  }

  /**
   * Actually determine whether python scripting is functional in LLDB. As there
   * are many reasons why python scripting would fail (e.g. disabled in CMake,
   * wrong SDK installation, etc.), it's more effective to just execute a
   * minimal script to confirm it's operative.
   */
  private async doLLDBHasPythonScriptingSupport(): Promise<boolean> {
    try {
      let { stdout, stderr } = await execFile(this.lldbPath, [
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
          `Python scripting support in LLDB not found. The test script returned:\n${stdout
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

  private constructor(
    loggingService: LoggingService,
    version: MojoSDKVersion,
    modularPath: string,
    rawConfig: { [key: string]: any }
  ) {
    this.loggingService = loggingService;

    this.version = version;
    this.modularHomePath = modularPath;
    this.mojoLLDBVSCodePath = rawConfig.lldb_vscode_path;
    this.mojoLLDBVisualizersPath = rawConfig.lldb_visualizers_path;
    this.mojoDriverPath = rawConfig.driver_path;
    this.mojoLanguageServerPath = rawConfig.lsp_server_path;
    this.mojoLLDBPluginPath = rawConfig.lldb_plugin_path;
    this.lldbPath = rawConfig.lldb_path;
  }

  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  /**
   * The version of the SDK.
   */
  version: MojoSDKVersion;

  /**
   * The MODULAR_HOME path containing the SDK.
   */
  modularHomePath: string = '';

  /**
   * The path to the mojo driver within the SDK installation.
   */
  mojoDriverPath: string = '';

  /**
   * The path to the LLDB vscode debug adapter.
   */
  mojoLLDBVSCodePath: string = '';

  /**
   * The path to the LLDB visualizers.
   */
  mojoLLDBVisualizersPath: string = '';

  /**
   * The path the mojo language server within the SDK installation.
   */
  mojoLanguageServerPath: string = '';

  /**
   * The path to the mojo LLDB plugin.
   */
  mojoLLDBPluginPath: string = '';

  /**
   * The path to the LLDB binary.
   */
  lldbPath: string = '';

  /**
   * A promise for if the LLDB binary has python scripting support.
   */
  private lldbHasPythonScriptingSupportResult?: Promise<boolean>;
}
