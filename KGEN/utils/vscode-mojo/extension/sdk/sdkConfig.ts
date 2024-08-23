//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { LoggingService } from '../logging';
import { MojoSDKVersion } from './sdkVersion';
import * as util from 'util';
const execFile = util.promisify(require('child_process').execFile);

/**
 * This class represents a subset of the Modular config object used by extension
 * for interacting with mojo. It should be handled a POD object.
 */
export class MojoSDKConfig {
  /**
   * The version of the SDK.
   */
  readonly version: MojoSDKVersion;

  /**
   * The MODULAR_HOME path containing the SDK.
   */
  readonly modularHomePath: string;

  /**
   * The path to the mojo driver within the SDK installation.
   */
  readonly mojoDriverPath: string;

  /**
   * The path to the LLDB vscode debug adapter.
   */
  readonly mojoLLDBVSCodePath: string;

  /**
   * The path to the LLDB visualizers.
   */
  readonly mojoLLDBVisualizersPath: string;

  /**
   * The path the mojo language server within the SDK installation.
   */
  readonly mojoLanguageServerPath: string;

  /**
   * The path to the mojo LLDB plugin.
   */
  readonly mojoLLDBPluginPath: string;

  /**
   * The path to the LLDB binary.
   */
  readonly lldbPath: string;

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
    return new MojoSDKConfig(version, modularPath, rawConfig);
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

  private constructor(
    version: MojoSDKVersion,
    modularPath: string,
    rawConfig: { [key: string]: any }
  ) {
    this.version = version;
    this.modularHomePath = modularPath;
    this.mojoLLDBVSCodePath = rawConfig.lldb_vscode_path;
    this.mojoLLDBVisualizersPath = rawConfig.lldb_visualizers_path;
    this.mojoDriverPath = rawConfig.driver_path;
    this.mojoLanguageServerPath = rawConfig.lsp_server_path;
    this.mojoLLDBPluginPath = rawConfig.lldb_plugin_path;
    this.lldbPath = rawConfig.lldb_path;
  }
}
