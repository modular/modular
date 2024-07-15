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

import { LoggingService } from './logging';
import * as config from './utils/config';
import { substituteVariables } from './utils/vscodeVariables';
import { isNightlyExtension } from './utils/buildInfo';
import { DisposableContext } from './utils/disposableContext';
import * as configWatcher from './utils/configWatcher';

/**
 * This class represents a Mojo SDK version.
 */
export class MojoSDKVersion {
  constructor(
    title: string,
    major: number,
    minor: number,
    patch: number,
    driverPath: string
  ) {
    this.title = title;
    this.minor = minor;
    this.major = major;
    this.patch = patch;
    this.driverPath = driverPath;
  }

  /**
   * Return if this is a dev version.
   */
  isDev(): boolean {
    return this.minor == 0 && this.major == 0 && this.patch == 0;
  }

  /**
   * Convert the version into a human readable string.
   */
  toString(): string {
    // If this is a dev build, format the title differently.
    if (this.isDev()) {
      // We include the path to the modular repo, which is three levels up from
      // the mojo driver path.
      const repo = path.join(path.parse(this.driverPath).dir, '..', '..', '..');
      return `${this.title} (dev) - ${repo}`;
    }

    // Otherwise, just format the version number.
    return `${this.title} (${this.major}.${this.minor}.${this.patch})`;
  }

  title: string;
  minor: number;
  major: number;
  patch: number;
  driverPath: string;
}

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

/**
 * Class used for interacting with and checking the status of the Mojo SDK.
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

    // Bail early if we don't have any keys.
    if (possibleKeys.length === 0) {
      return undefined;
    }

    // If this is a dev-build path, there'll only be one key so just grab
    // it.

    // If this is a dev-build path, there'll only be one key so just grab
    // it.

    // If this is a dev-build path, there'll only be one key so just grab
    // it.
    if (modularHomePath.endsWith('.derived')) {
      return possibleKeys[0];
    }

    // Filter the keys to only those that match the current extension.

    // Filter the keys to only those that match the current extension.
    possibleKeys = possibleKeys.filter(
      (key) => isNightly == key.endsWith('-nightly')
    );

    if (possibleKeys.length === 0) {
      return undefined;
    }

    // Prefer the 'max' key if it exists.

    // Prefer the 'max' key if it exists.
    const maxKey = possibleKeys.find((key) => key.includes('max'));

    if (maxKey) {
      return maxKey;
    }

    // Otherwise, just grab the first key.

    // Otherwise, just grab the first key.
    return possibleKeys[0];
  }

  /**
   * Emit a warning to the user if the current SDK is out of date.
   */
  public async warnIfSDKOutOfDate() {
    // If this is a dev-build, there's no version to check.

    // If this is a dev-build, there's no version to check.
    if (this.config.version.isDev()) {
      return;
    }

    // Grab the current extension version.

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
}

/**
 *  This class manages the resolution of SDKs for different files, workspaces
 * and tools.
 */
export class MojoSDKManager extends DisposableContext {
  /**
   * The main SDK owned by the manager.
   */
  private sdk: Promise<MojoSDK | undefined> | undefined;

  /**
   * A service that can be used to log message in the Mojo output channel.
   */
  private loggingService: LoggingService;

  /**
   * The current extension context.
   */
  private readonly context: vscode.ExtensionContext;

  constructor(
    loggingService: LoggingService,
    context: vscode.ExtensionContext
  ) {
    super();
    this.loggingService = loggingService;
    this.context = context;

    this.pushSubscription(
      vscode.commands.registerCommand('mojo.sdk.install', () => {
        this.promptInstallSDK();
      })
    );
  }

  /**
   * Resolve the Modular config for the extension.
   *
   * The resolver will look for available SDKs in a few specified locations:
   *   - The `mojo.modularHomePath` setting in the user settings, and any
   *     current workspaces.
   *   - The `MODULAR_HOME` environment variable.
   *   - The packages installed via the `modular` cli tool.
   *
   * If a single SDK is found, that is the SDK used for the extension. If
   * multiple are found, the user is prompted for which SDK they would like to
   * use.
   *
   * This function caches the result and the cache is refreshed whenever there's
   * a change in the list of active workspaces.
   */
  public async findSDK(): Promise<MojoSDK | undefined> {
    // Find the SDK if we haven't yet.
    if (!this.sdk) {
      this.sdk = this.doFindSDK();
      this.sdk.then((sdk) => {
        if (!sdk) {
          this.promptInstallSDK(/*notifySDKNotFound=*/ true);
        }
      });
    }

    return this.sdk;
  }

  /**
   * Finds all of the possible Mojo SDKs reachable by the extension. This checks
   * all of the possible locations as described by `findSDK`.
   */
  private async findAllPossibleSDKs(): Promise<MojoSDK[]> {
    // SDKs come from two possible places:
    //  * The `mojo.modularHomePath` setting, which should generally only be
    //  used
    //    in a dev build.
    //  * The `MODULAR_HOME` environment variable.
    //  * The configurations defined via the `modular` tool.
    let possibleSDKs: MojoSDK[] = [];

    // Utilities for processing SDKs found via modular home paths.
    let checkedPaths = new Set<string>();
    let addSDKPath = async (path: string | undefined) => {
      if (!path || checkedPaths.has(path)) {
        return;
      }
      checkedPaths.add(path);
      let sdk = await this.loadSDKFromModularHome(path);

      if (sdk) {
        possibleSDKs.push(sdk);
      }
    };

    // First, find the possible SDKs by looking at the `mojo.modularHomePath`
    // setting.
    if (vscode.workspace.workspaceFolders) {
      for (let workspaceFolder of vscode.workspace.workspaceFolders) {
        await addSDKPath(
          await this.tryGetModularHomePathFromConfig(workspaceFolder)
        );
      }
    }
    await addSDKPath(await this.tryGetModularHomePathFromConfig(undefined));

    // Next, check the `MODULAR_HOME` environment variable.
    await addSDKPath(process.env.MODULAR_HOME);

    // Finally, check the configurations defined via the `modular` tool.
    possibleSDKs.push(...(await this.findPossibleSDKsFromCLI()));

    /// Remove duplicate SDKs (as determined by version).
    let seenVersions = new Set<string>();
    return possibleSDKs.filter((sdk) => {
      let version = sdk.config.version.toString();

      if (seenVersions.has(version)) {
        return false;
      }
      seenVersions.add(version);
      return true;
    });
  }

  /**
   * Find all of the viable Mojo SDKs installed via the `modular` cli tool.
   */
  private async findPossibleSDKsFromCLI(): Promise<MojoSDK[]> {
    let isNightly = isNightlyExtension(this.context);

    // Build a regex to match an .ini like string, where the form is:
    //   section.key = value
    // the section must start with `mojo`.
    let valueRegex = new RegExp(`^(mojo[^.]*)\\.([^.]+) = ([^;]*);?$`);

    // The first step is to invoke the `modular` cli and collect all of the
    // mojo related configuration values, bucketing them by the top-level
    // section.
    let configurationValues = new Map<string, { [key: string]: any }>();
    try {
      let { stdout, stderr } = await execFile('modular', ['config-list']);
      for (let line of stdout.split('\n')) {
        line = line.trim();

        // Match the value regex.
        let match = valueRegex.exec(line);

        if (!match) {
          continue;
        }
        let section = match[1];
        let key = match[2];
        let value = match[3];

        // Ignore nightly configs in non-nightly extensions, and vice versa.

        // Ignore nightly configs in non-nightly extensions, and vice versa.
        if (isNightly != section.endsWith('-nightly')) {
          continue;
        }

        // Set this configuration value.

        // Set this configuration value.

        // Set this configuration value.
        if (!configurationValues.has(section)) {
          configurationValues.set(section, {});
        }
        configurationValues.get(section)![key] = value;
      }
    } catch (e) {
      this.loggingService.main.logError(
        'Unable to invoke `modular config-list`, failed with: ',
        e
      );
    }

    // Build a possible SDK for each of the configurations.
    let possibleSDKs: MojoSDK[] = [];
    for (let [section, values] of configurationValues) {
      let sdk = await this.createSDKAndConfig(section, values);

      if (sdk) {
        possibleSDKs.push(sdk);
      }
    }
    return possibleSDKs;
  }

  /**
   * Load a Mojo SDK defined at the given modular home location.
   */
  private async loadSDKFromModularHome(
    modularPath: string
  ): Promise<MojoSDK | undefined> {
    this.loggingService.main.logInfo(`MODULAR_HOME is ${modularPath}.`);

    // Read in the config file.
    const modularCfg = path.join(modularPath, 'modular.cfg');
    let configPath = vscode.Uri.file(modularCfg);

    try {
      let configPathStat = await vscode.workspace.fs.stat(configPath);
      if (!(configPathStat.type & vscode.FileType.File)) {
        this.showSDKErrorMessage(
          `The modular config file '${modularCfg}' is not a file.`
        );
        return undefined;
      }
    } catch (e) {
      this.showSDKErrorMessage(
        `The modular config file '${
          modularCfg
        }' does not exist or VS Code does not have permissions to access it.`,
        e
      );
      return undefined;
    }
    let modularConfig = ini.parse(
      new TextDecoder().decode(await vscode.workspace.fs.readFile(configPath))
    );
    this.loggingService.main.logInfo(
      'modular.cfg file with contents',
      modularConfig
    );

    // Find the appropriate mojo configuration key in the config file.
    let mojoKeys: string[] = Object.keys(modularConfig).filter((key) =>
      key.startsWith('mojo')
    );
    let configKey = MojoSDK.getConfigKey(
      modularPath,
      isNightlyExtension(this.context),
      mojoKeys
    );
    if (!configKey) {
      this.showSDKErrorMessage(
        `The modular config file '${modularCfg}' is outdated.`
      );
      return undefined;
    }

    return this.createSDKAndConfig(
      configKey,
      modularConfig[configKey],
      modularPath
    );
  }

  /**
   * Create a Mojo SDK from the given configuration.
   */
  private async createSDKAndConfig(
    configSection: string,
    rawConfig: { [key: string]: any },
    modularPath: string = ''
  ): Promise<MojoSDK | undefined> {
    let sdkConfig = await MojoSDKConfig.create(
      this.loggingService,
      modularPath,
      configSection,
      rawConfig
    );

    if (!sdkConfig) {
      return undefined;
    }
    return new MojoSDK(sdkConfig, this.loggingService, this.context);
  }

  /**
   * This function discovers an SDK following the procedure described in
   * `findSDK`.
   */
  private async doFindSDK(): Promise<MojoSDK | undefined> {
    // Find the possible set of SDKs.
    let possibleSDKs = await this.findAllPossibleSDKs();

    if (possibleSDKs.length == 0) {
      return undefined;
    }

    // Resolve the SDK from the set of possible choices.

    // Resolve the SDK from the set of possible choices.
    let sdk = possibleSDKs[0];
    if (possibleSDKs.length > 1) {
      // If there are multiple, ask the user which one they want to use.
      let sdkNames = possibleSDKs.map((sdk) => sdk.config.version.toString());
      let selected = await vscode.window.showQuickPick(sdkNames, {
        placeHolder: 'Select the Mojo SDK to use!',
        ignoreFocusOut: true,
      });
      if (selected) {
        sdk = possibleSDKs.find(
          (sdk) => sdk.config.version.toString() == selected
        )!;
      }
    }

    // Push a subscription for changes to any of the SDK paths.
    this.pushSubscription(
      await configWatcher.activate(
        undefined,
        [],
        [
          sdk.config.mojoLLDBVSCodePath,
          sdk.config.mojoDriverPath,
          sdk.config.mojoLanguageServerPath,
          sdk.config.mojoLLDBPluginPath,
          sdk.config.lldbPath,
        ]
      )
    );

    // Now that we have a resolved SDK, warn if it's out of date.
    await sdk.warnIfSDKOutOfDate();
    return sdk;
  }

  /**
   * Prompt to the user that the SDK is missing, and provide a link to the
   * installation instructions.
   */
  private async promptInstallSDK(notifySDKNotFound: boolean = false) {
    this.loggingService.main.logInfo('Prompting Install SDK.');
    const prefix = notifySDKNotFound
      ? 'The Mojo🔥 development environment was not found. '
      : '';

    let value = await vscode.window.showInformationMessage(
      prefix +
        'If the Mojo SDK is installed, please set the MODULAR_HOME environment variable to the ' +
        'appropriate path, or set the `mojo.modularHomePath` configuration. If you do ' +
        'not have it installed, would you like to install it?',
      'Install',
      'Open setting'
    );
    if (value === 'Install') {
      // TODO: This should resolve to the actual mojo download link when
      // the user console is in place.
      vscode.env.openExternal(vscode.Uri.parse('https://www.modular.com/mojo'));
    } else if (value === 'Open setting') {
      vscode.commands.executeCommand('workbench.action.openGlobalSettings', {
        openToSide: false,
        query: `mojo.modularHomePath`,
      });
    }
  }

  /**
   * Attempt to retrieve the modular home path from the config. This will also
   * perform the substitution of some common VSCode variables.
   *
   * If the setting does not exist or the resolved path is not a directory,
   * return undefined.
   */
  private async tryGetModularHomePathFromConfig(
    workspaceFolder: vscode.WorkspaceFolder | undefined
  ): Promise<string | undefined> {
    let modularPath = config.get<string>('modularHomePath', workspaceFolder);

    if (!modularPath) {
      return undefined;
    }
    const substituted = substituteVariables(modularPath, workspaceFolder);

    const showError = (reason: string) => {
      let message = `The mojo.modularHomePath setting '${modularPath}'`;

      if (substituted !== modularPath) {
        message += `, which resolves to '${substituted}',`;
      }
      message += ' ' + reason + '.';
      this.showSDKErrorMessage(message);
      return undefined;
    };

    if (substituted.length == 0) {
      return showError('is empty');
    }

    try {
      let configPathStat = await vscode.workspace.fs.stat(
        vscode.Uri.file(substituted)
      );

      if (configPathStat.type & vscode.FileType.Directory) {
        return substituted;
      }
      return showError('is not a directory');
    } catch (err) {
      return showError('does not exist');
    }
  }

  /**
   * Show an error message as a VSCode notification and log it to the output
   * channel as well.
   */
  private showSDKErrorMessage(message: string, error?: unknown): void {
    message = 'Mojo SDK initialization error: ' + message;
    this.loggingService.main.logError(message, error);
    vscode.window.showErrorMessage(message);
  }
}
