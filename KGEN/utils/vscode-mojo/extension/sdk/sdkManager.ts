//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as ini from 'ini';
import * as path from 'path';
import * as util from 'util';
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as config from '../utils/config';

const execFile = util.promisify(require('child_process').execFile);
const chmod = util.promisify(require('fs').chmod);

import { Logger } from '../logging';
import { isNightlyExtension } from '../utils/buildInfo';
import { DisposableContext } from '../utils/disposableContext';
import { lock } from 'proper-lockfile';
import { MojoSDKConfig } from './sdkConfig';
import { MojoSDK } from './sdk';
import { Mutex } from 'async-mutex';
import {
  directoryExists,
  fileExists,
  getAllOpenMojoFiles,
  mkdirp,
  moveUpUntil,
  readFile,
} from '../utils/files';
import { MojoSDKVersion } from './sdkVersion';
import axios from 'axios';

export type MojoSDKSpec = {
  kind: 'modular-cli' | 'dev' | 'magic';
  modularHomePath: string;
  section: string;
  version: MojoSDKVersion;
};

type NotYetSelectedSDK = {
  state: 'not-yet-selected';
};

type SelectedSDK = {
  state: 'selected';
  sdkSpec: Optional<MojoSDKSpec>;
  errorMessage?: Optional<string>;
};

type SDKSelection = NotYetSelectedSDK | SelectedSDK;

/**
 * This class manages the active SDK, switching SDKs, and other related ad hoc actions.
 *
 * There are two public APIs:
 *  - `findSDK` is the way to get the active SDK and it's protected by a mutex.
 *  - `createAdHocSDKAndShowError` is used for actions that force the use of a given SDK.
 *    This function doesn't have side effects.
 *
 * Caching should be minimized to capture the current state of the SDKs in the filesystem.
 */
export class MojoSDKManager extends DisposableContext {
  public logger: Logger;
  private context: vscode.ExtensionContext;
  private initializationSDK: Optional<MojoSDKSpec>;
  private enableMagicSDK: boolean;
  private activeSDK: SDKSelection = { state: 'not-yet-selected' };
  private findSDKMutex = new Mutex();
  private isNightly: boolean;

  constructor(
    logger: Logger,
    context: vscode.ExtensionContext,
    initializationSDK: Optional<MojoSDKSpec>,
    enableMagicSDK: boolean
  ) {
    super();
    this.logger = logger;
    this.context = context;
    this.initializationSDK = initializationSDK;
    this.enableMagicSDK = enableMagicSDK;
    this.isNightly = isNightlyExtension(this.context);
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.sdk.selectSdk', async () => {
        const allSDKSpecs = await this.findAllSDKs();
        if (allSDKSpecs.length === 0) {
          vscode.window.showErrorMessage('No MAX SDKs were found.');
          return;
        }
        const sdkNames = allSDKSpecs.map((spec) => spec.version.toString());
        const selected = await vscode.window.showQuickPick(sdkNames, {
          ignoreFocusOut: true,
          title: 'Select the Max SDK to use',
        });
        const selectedSDK = allSDKSpecs.find(
          (spec) => spec.version.toString() == selected
        );
        if (selectedSDK !== undefined) {
          vscode.commands.executeCommand('mojo.restart', selectedSDK);
        }
      })
    );
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.magicSdk.install', async () => {
        const spec = await this.findMagicSDKSpec(/*withLock=*/ false);
        if (spec !== undefined) {
          vscode.commands.executeCommand('mojo.restart');
        }
      })
    );
  }

  public async findSDK(
    hideRepeatedErrors: boolean
  ): Promise<Optional<MojoSDK>> {
    const doWork = async () => {
      if (this.activeSDK.state === 'selected') {
        return this.createSDKAndShowError(this.activeSDK, hideRepeatedErrors);
      } else {
        const activeSDK = await this.initializeActiveSDK();
        return this.createSDKAndShowError(
          activeSDK,
          /*hideRepeatedErrors=*/ false
        );
      }
    };

    return this.findSDKMutex.runExclusive(() => doWork());
  }

  public async createAdHocSDKAndShowError(
    modularHomePath: string,
    section?: string
  ): Promise<Optional<MojoSDK>> {
    const hideRepeatedErrors = false;

    const devSDKSpec = await this.findDevSDKSpecFromSubPath(modularHomePath);
    if (devSDKSpec !== undefined) {
      return this.createSDKAndShowError(
        { state: 'selected', sdkSpec: devSDKSpec },
        hideRepeatedErrors
      );
    }
    if (
      this.activeSDK.state === 'selected' &&
      this.activeSDK.sdkSpec?.modularHomePath === modularHomePath
    ) {
      return this.createSDKAndShowError(this.activeSDK, hideRepeatedErrors);
    }
    // TODO: create an SDK from a config file and a section once every debug request has a section
    vscode.window.showErrorMessage(
      'Unable to determine the SDK for ' + modularHomePath
    );
    return undefined;
  }

  private async initializeActiveSDK(): Promise<SelectedSDK> {
    // This is invoked only once per extension activation.
    const sdkSpec =
      this.initializationSDK !== undefined
        ? this.initializationSDK
        : await this.selectSDK();
    this.activeSDK = { state: 'selected', sdkSpec };
    return this.activeSDK;
  }

  private async createSDKAndShowError(
    selectedSDK: SelectedSDK,
    hideRepeatedErrors: boolean
  ): Promise<Optional<MojoSDK>> {
    const result = await this.doCreateSDK(selectedSDK);
    if (typeof result === 'string') {
      if (hideRepeatedErrors && selectedSDK.errorMessage === result) {
        return undefined;
      }
      let errorMessage = result;
      selectedSDK.errorMessage = result;

      if (
        selectedSDK.sdkSpec?.kind == 'modular-cli' ||
        selectedSDK.sdkSpec === undefined
      ) {
        errorMessage += '\nPlease install the MAX SDK via the modular tool.';
        vscode.window
          .showErrorMessage(errorMessage, 'Install')
          .then((value) => {
            if (value === 'Install') {
              vscode.env.openExternal(
                vscode.Uri.parse('https://www.modular.com/mojo')
              );
            }
          });
      } else if (selectedSDK.sdkSpec.kind === 'dev') {
        errorMessage += '\nPlease run ./bazelw run //:install.';
        vscode.window
          .showErrorMessage(errorMessage, 'Run bazel')
          .then((value) => {
            if (value === 'Run bazel') {
              const repo = path.dirname(
                selectedSDK.sdkSpec?.modularHomePath || ''
              );
              const terminal =
                vscode.window.activeTerminal ||
                vscode.window.createTerminal({
                  name: repo,
                });
              terminal.sendText(`(cd '${repo}' && ./bazelw run //:install)`);
            }
          });
      } else if (selectedSDK.sdkSpec.kind === 'magic') {
        errorMessage += '\nPlease reinstall the MAX SDK for VS Code.';
        vscode.window
          .showErrorMessage(errorMessage, 'Reinstall')
          .then(async (value) => {
            if (value === 'Reinstall') {
              vscode.commands.executeCommand('mojo.magicSdk.install');
            }
          });
      }
      this.logger.main.logError(errorMessage);
      return undefined;
    }
    return result;
  }

  private async doCreateSDK(
    selectedSDK: SelectedSDK
  ): Promise<MojoSDK | string> {
    const spec = selectedSDK.sdkSpec;
    if (spec === undefined) {
      return 'The Mojo🔥 development environment was not found.';
    }
    const modularConfigPath = path.join(spec.modularHomePath, 'modular.cfg');
    const modularConfigContents = await readFile(modularConfigPath);
    if (modularConfigContents === undefined) {
      return `The modular config file '${modularConfigPath}' can't be read.`;
    }
    const modularConfig = ini.parse(modularConfigContents);
    this.logger.main.logInfo(
      `'${modularConfigPath}' with contents`,
      modularConfig
    );

    let sdkConfig = await MojoSDKConfig.create(
      this.logger,
      spec.modularHomePath,
      spec.section,
      modularConfig[spec.section]
    );

    if (!sdkConfig) {
      return `Unable to determine the MAX SDK version.`;
    }
    return new MojoSDK(sdkConfig, this.logger, this.context);
  }

  private async selectSDK(): Promise<Optional<MojoSDKSpec>> {
    const allSDKSpecs = await this.findAllSDKs();
    if (allSDKSpecs.length === 0) {
      return undefined;
    }
    if (allSDKSpecs.length === 1) {
      return allSDKSpecs[0];
    }
    const sdkNames = allSDKSpecs.map((spec) => spec.version.toString());
    const selected = await vscode.window.showQuickPick(sdkNames, {
      ignoreFocusOut: true,
      title: 'Select the Max SDK to use',
    });
    return allSDKSpecs.find((spec) => spec.version.toString() == selected);
  }

  private async findAllSDKs(): Promise<MojoSDKSpec[]> {
    const [devSDKSpecs, releaseSDKSpecs] = await Promise.all([
      this.findDevSDKSpecs(),
      this.findReleaseSDKSpecs(),
    ]);

    return [...devSDKSpecs, ...releaseSDKSpecs];
  }

  private async findDevSDKSpecs(): Promise<MojoSDKSpec[]> {
    const visiblePaths = [];
    const [activeMojoFile, otherOpenMojoFiles] = getAllOpenMojoFiles();

    if (activeMojoFile) {
      visiblePaths.push(activeMojoFile.uri.fsPath);
    }
    for (const file of otherOpenMojoFiles) {
      visiblePaths.push(file.uri.fsPath);
    }
    for (let workspaceFolder of vscode.workspace.workspaceFolders || []) {
      visiblePaths.push(workspaceFolder.uri.fsPath);
    }
    const candidateSDKSpecs = (
      await Promise.all(
        visiblePaths.map((path) => this.findDevSDKSpecFromSubPath(path))
      )
    ).filter((x): x is MojoSDKSpec => x !== undefined);
    const uniqueSDKSpecs = new Map<string, MojoSDKSpec>();
    candidateSDKSpecs.forEach((spec) =>
      uniqueSDKSpecs.set(spec.modularHomePath, spec)
    );
    return [...uniqueSDKSpecs.values()];
  }

  private async findDevSDKSpecFromSubPath(
    fsPath: string
  ): Promise<Optional<MojoSDKSpec>> {
    const repoRoot = await moveUpUntil(fsPath, (p) =>
      directoryExists(path.join(p, '.git'))
    );
    if (!repoRoot) {
      return undefined;
    }
    const bazelPath = path.join(repoRoot, 'WORKSPACE.bazel');
    const bazelBytes = await vscode.workspace.fs.readFile(
      vscode.Uri.file(bazelPath)
    );
    const bazelContents = Buffer.from(bazelBytes).toString('utf-8');
    if (!bazelContents.includes('workspace(name = "modular")')) {
      return undefined;
    }
    const modularHomePath = path.join(repoRoot, '.derived');
    return {
      kind: 'dev',
      modularHomePath,
      version: new MojoSDKVersion(
        'Modular Repo',
        '0',
        '0',
        '0',
        modularHomePath
      ),
      section: 'mojo-max',
    };
  }

  private async findReleaseSDKSpecs(): Promise<MojoSDKSpec[]> {
    if (this.enableMagicSDK) {
      const spec = await this.findMagicSDKSpec(/*withLock=*/ true);
      return spec ? [spec] : [];
    }
    return this.findModularCliSDKSpecs();
  }

  private async downloadFile(url: string, outputPath: string) {
    const writer = fs.createWriteStream(outputPath);

    const response = await axios({
      url,
      method: 'GET',
      responseType: 'stream',
    });

    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  }

  private async findMagicSDKSpec(
    withLock: boolean
  ): Promise<Optional<MojoSDKSpec>> {
    const privateDir = this.context.globalStorageUri.fsPath;
    const magicDataHome = path.join(privateDir, 'magic-data-home');
    await mkdirp(magicDataHome);

    const magicPath = path.join(privateDir, 'magic');
    const doneDirectory = path.join(privateDir, 'done');

    let platform: string;
    if (process.platform === 'linux') {
      platform = 'unknown-linux-musl';
    } else if (process.platform === 'darwin') {
      platform = 'apple-darwin';
    } else if (process.platform === 'win32') {
      platform = 'pc-windows-msvc';
    } else {
      vscode.window.showErrorMessage(
        `The MAX SDK is not supported in this platform: ${process.platform}`
      );
      return undefined;
    }
    let arch: string;
    if (process.arch === 'x64') {
      arch = 'x86_64';
    } else if (process.arch === 'arm64') {
      arch = 'aarch64';
    } else {
      arch = process.arch;
    }
    const magicUrl = `https://dl.modular.com/public/magic/raw/versions/latest/magic-${arch}-${platform}`;
    const major = config.get(
      'magicSDK.major',
      /*workspaceFolder=*/ undefined,
      ''
    );
    const minor = config.get(
      'magicSDK.minor',
      /*workspaceFolder=*/ undefined,
      ''
    );
    const patch = config.get(
      'magicSDK.patch',
      /*workspaceFolder=*/ undefined,
      ''
    );
    const version = `${major}.${minor}.${patch}`;

    const success = await vscode.window.withProgress<boolean>(
      {
        title: 'Installing the MAX SDK for VS Code',
        location: vscode.ProgressLocation.Notification,
      },
      async (): Promise<boolean> => {
        try {
          this.logger.main.logInfo('Trying to acquire installation lock...');
          const release = withLock
            ? await lock(privateDir, { retries: 10 })
            : async () => {};
          const versionDoneDir = path.join(privateDir, version);

          if (
            (await directoryExists(doneDirectory)) &&
            (await directoryExists(versionDoneDir))
          ) {
            this.logger.main.logInfo(
              'Magic SDK present. Skipping installation.'
            );
            await release();
            return true;
          }
          fs.rmdirSync(doneDirectory);

          this.logger.main.logInfo(
            `Will download ${magicUrl} into ${magicPath}`
          );
          await this.downloadFile(magicUrl, magicPath);
          this.logger.main.logInfo('Successfully downloaded');
          await chmod(magicPath, 0o755);
          this.logger.main.logInfo(
            `The permissions for ${magicPath} have been changed.`
          );

          this.logger.main.logInfo(`Will install MAX`);
          const env = { ...process.env };
          env['MAGIC_DATA_HOME'] = magicDataHome;

          const args = [
            'global',
            'install',
            '-c',
            'https://conda.modular.com/max',
            '-c',
            'conda-forge',
            `max==${version}`,
            'python>=3.11,<3.12',
          ];
          await execFile(magicPath, args, { env });
          this.logger.main.logInfo(`Successfully installed MAX`);

          await mkdirp(doneDirectory);
          await mkdirp(versionDoneDir);
          await release();
          return true;
        } catch (e) {
          this.logger.main.logError(
            "Couldn't install the MAX SDK for VS Code",
            e
          );
          return false;
        }
      }
    );
    if (!success) {
      vscode.window.showErrorMessage(
        "Couldn't install the MAX SDK for VS Code"
      );
      return undefined;
    }
    const modularHomePath = path.join(
      magicDataHome,
      'envs',
      'max',
      'share',
      'max'
    );

    // Consider nightly here
    return {
      kind: 'magic',
      modularHomePath,
      section: 'mojo-max',
      version: new MojoSDKVersion(
        'MAX SDK for VS Code',
        major,
        minor,
        patch,
        modularHomePath
      ),
    };
  }

  private async findModularCliSDKSpecs(): Promise<MojoSDKSpec[]> {
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
        if (this.isNightly != section.endsWith('-nightly')) {
          continue;
        }

        // Set this configuration value.
        if (!configurationValues.has(section)) {
          configurationValues.set(section, {});
        }
        configurationValues.get(section)![key] = value;
      }
    } catch (e) {
      this.logger.main.logError(
        'Unable to invoke `modular config-list`, failed with: ',
        e
      );
    }

    // Build a possible SDK for each of the configurations.
    let possibleSDKs: MojoSDKSpec[] = [];
    for (let [section, values] of configurationValues) {
      const mojoPath = values.driver_path || '';
      const modularHomePath = await moveUpUntil(mojoPath, async (p) => {
        const [d1, d2] = await Promise.all([
          directoryExists(path.join(p, 'pkg')),
          fileExists(path.join(p, 'modular.cfg')),
        ]);
        return d1 && d2;
      });
      if (modularHomePath !== undefined) {
        const version = await MojoSDKConfig.parseVersionFromDriver(
          this.logger,
          mojoPath,
          section
        );
        if (version !== undefined) {
          possibleSDKs.push({
            kind: 'modular-cli',
            modularHomePath,
            section,
            version,
          });
        }
      }
    }
    return possibleSDKs;
  }
}
