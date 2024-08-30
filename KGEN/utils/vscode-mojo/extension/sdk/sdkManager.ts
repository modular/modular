//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as ini from 'ini';
import * as path from 'path';
import * as util from 'util';
import * as vscode from 'vscode';
import { Logger } from '../logging';
import { DisposableContext } from '../utils/disposableContext';
import { MojoSDKConfig } from './sdkConfig';
import { MojoSDK } from './sdk';
import { Mutex } from 'async-mutex';
import {
  directoryExists,
  fileExists,
  getAllOpenMojoFiles,
  moveUpUntil,
  readFile,
} from '../utils/files';
import { MojoSDKVersion } from './sdkVersion';
import { findMagicSDKSpec } from './magicSdk';
import { MojoSDKSpec } from './types';
import { ExtensionSemiPersistentState } from '../extension';
const execFile = util.promisify(require('child_process').execFile);

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
  private extensionSemiPersistentState;

  constructor(
    logger: Logger,
    context: vscode.ExtensionContext,
    initializationSDK: Optional<MojoSDKSpec>,
    isNightly: boolean,
    enableMagicSDK: boolean,
    extensionSemiPersistentState: ExtensionSemiPersistentState
  ) {
    super();
    this.logger = logger;
    this.context = context;
    this.initializationSDK = initializationSDK;
    this.enableMagicSDK = enableMagicSDK;
    this.isNightly = isNightly;
    this.extensionSemiPersistentState = extensionSemiPersistentState;

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
          placeHolder:
            this.activeSDK.state === 'selected' && this.activeSDK.sdkSpec
              ? `Currently using ${this.activeSDK.sdkSpec.version.toString()}`
              : 'Select an SDK or cancel',
        });
        const selectedSDK = allSDKSpecs.find(
          (spec) => spec.version.toString() === selected
        );
        if (selectedSDK !== undefined) {
          vscode.commands.executeCommand('mojo.restart', selectedSDK);
        }
      })
    );
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.magicSdk.install', async () => {
        const spec = await findMagicSDKSpec(
          /*withLock=*/ false,
          this.context,
          this.logger,
          this.isNightly,
          /*reinstall=*/ true
        );
        if (spec !== undefined) {
          vscode.commands.executeCommand('mojo.restart');
        }
      })
    );
    this.pushSubscription(
      vscode.workspace.onDidOpenTextDocument((doc: vscode.TextDocument) =>
        this.onPathSeenAfterInitialization(doc.uri.fsPath)
      )
    );
    this.pushSubscription(
      vscode.workspace.onDidOpenNotebookDocument(
        (doc: vscode.NotebookDocument) =>
          this.onPathSeenAfterInitialization(doc.uri.fsPath)
      )
    );
    this.pushSubscription(
      vscode.workspace.onDidChangeWorkspaceFolders(
        (e: vscode.WorkspaceFoldersChangeEvent) => {
          for (const added of e.added) {
            this.onPathSeenAfterInitialization(added.uri.fsPath);
          }
        }
      )
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

  private async onPathSeenAfterInitialization(path: string): Promise<void> {
    const specResult = await this.findDevSDKSpecFromSubPath(path);
    if (specResult !== undefined && specResult.isNew) {
      const result = await vscode.window.showInformationMessage(
        'A new MAX SDK from a Modular repo has been identified: ' +
          specResult.spec.modularHomePath +
          '.\nDo you want to use this SDK?',
        'Use this SDK'
      );
      if (result === 'Use this SDK') {
        vscode.commands.executeCommand('mojo.restart', specResult.spec);
      }
    }
  }

  public async createAdHocSDKAndShowError(
    modularHomePath: string,
    section: Optional<string>
  ): Promise<Optional<MojoSDK>> {
    const hideRepeatedErrors = false;

    const devSDKSpec = await this.findDevSDKSpecFromSubPath(modularHomePath);
    if (devSDKSpec !== undefined) {
      return this.createSDKAndShowError(
        { state: 'selected', sdkSpec: devSDKSpec.spec },
        hideRepeatedErrors
      );
    }
    if (
      this.activeSDK.state === 'selected' &&
      this.activeSDK.sdkSpec?.modularHomePath === modularHomePath &&
      this.activeSDK.sdkSpec.section === section
    ) {
      return this.createSDKAndShowError(this.activeSDK, hideRepeatedErrors);
    }
    const sdkSpec: MojoSDKSpec = {
      kind: 'custom',
      modularHomePath,
      section: section || 'mojo-max' + (this.isNightly ? '-nightly' : ''),
      version: new MojoSDKVersion(
        modularHomePath,
        '0',
        '0',
        '0',
        modularHomePath
      ),
    };
    return this.createSDKAndShowError(
      { state: 'selected', sdkSpec },
      hideRepeatedErrors
    );
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

      if (selectedSDK.sdkSpec?.kind === 'modular-cli') {
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
      } else if (selectedSDK.sdkSpec?.kind === 'dev') {
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
      } else if (selectedSDK.sdkSpec?.kind === 'magic') {
        errorMessage += '\nPlease reinstall the MAX SDK for VS Code.';
        vscode.window
          .showErrorMessage(errorMessage, 'Reinstall')
          .then(async (value) => {
            if (value === 'Reinstall') {
              vscode.commands.executeCommand('mojo.magicSdk.install');
            }
          });
      } else if (selectedSDK.sdkSpec?.kind === 'custom') {
        errorMessage += `\nPlease reinstall or rebuild the ${selectedSDK.sdkSpec.section} SDK given by ${selectedSDK.sdkSpec.modularHomePath}.`;
        vscode.window.showErrorMessage(errorMessage);
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
    const mojoConfig = modularConfig[spec.section];
    if (!mojoConfig) {
      return `The modular config file '${modularConfigPath}' doesn't have the expected section ${spec.section}`;
    }
    const sdkConfig = new MojoSDKConfig(
      spec.version,
      spec.modularHomePath,
      mojoConfig
    );
    if (!sdkConfig) {
      return `Unable to determine the MAX SDK version.`;
    }
    return new MojoSDK(sdkConfig, this.logger);
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
    const selected =
      (await vscode.window.showQuickPick(sdkNames, {
        ignoreFocusOut: true,
        title: 'Select the Max SDK to use',
        placeHolder:
          'Select an SDK or cancel to select the first one in the list',
      })) || sdkNames[0];
    return allSDKSpecs.find((spec) => spec.version.toString() === selected);
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
    return this.findDevSDKSpecsFromSubPaths(visiblePaths);
  }

  private async findDevSDKSpecsFromSubPaths(
    paths: string[]
  ): Promise<MojoSDKSpec[]> {
    const candidateSDKSpecs = (
      await Promise.all(
        paths.map((path) => this.findDevSDKSpecFromSubPath(path))
      )
    )
      .map((x) => x?.spec)
      .filter((x): x is MojoSDKSpec => x !== undefined);
    const uniqueSDKSpecs = new Map<string, MojoSDKSpec>();
    candidateSDKSpecs.forEach((spec) =>
      uniqueSDKSpecs.set(spec.modularHomePath, spec)
    );
    return [...uniqueSDKSpecs.values()];
  }

  private async findDevSDKSpecFromSubPath(
    fsPath: string
  ): Promise<Optional<{ spec: MojoSDKSpec; isNew: boolean }>> {
    const repoRoot = await moveUpUntil(fsPath, (p) =>
      directoryExists(path.join(p, '.git'))
    );
    if (!repoRoot) {
      return undefined;
    }
    const bazelPath = path.join(repoRoot, 'WORKSPACE.bazel');
    try {
      const bazelBytes = await vscode.workspace.fs.readFile(
        vscode.Uri.file(bazelPath)
      );
      const bazelContents = Buffer.from(bazelBytes).toString('utf-8');
      if (!bazelContents.includes('workspace(name = "modular")')) {
        return undefined;
      }
      const modularHomePath = path.join(repoRoot, '.derived');
      const spec: MojoSDKSpec = {
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
      let isNew = false;
      if (
        !this.extensionSemiPersistentState.seenDevSDKs.has(spec.modularHomePath)
      ) {
        this.extensionSemiPersistentState.seenDevSDKs.add(spec.modularHomePath);
        isNew = true;
      }
      return { spec, isNew };
    } catch {
      return undefined;
    }
  }

  private async findReleaseSDKSpecs(): Promise<MojoSDKSpec[]> {
    // In tests, we don't want to download magic and its SDK.
    if (this.context.extensionMode === vscode.ExtensionMode.Test) {
      return [];
    }
    if (this.enableMagicSDK) {
      const spec = await findMagicSDKSpec(
        /*withLock=*/ true,
        this.context,
        this.logger,
        this.isNightly
      );
      return spec ? [spec] : [];
    }
    return this.findModularCliSDKSpecs();
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
          if (
            possibleSDKs.find(
              (sdk) =>
                sdk.modularHomePath === modularHomePath &&
                sdk.version.major === version.major &&
                sdk.version.minor === version.minor &&
                sdk.version.patch === version.patch
            ) === undefined
          ) {
            possibleSDKs.push({
              kind: 'modular-cli',
              modularHomePath,
              section,
              version,
            });
          }
        }
      }
    }
    return possibleSDKs;
  }
}
