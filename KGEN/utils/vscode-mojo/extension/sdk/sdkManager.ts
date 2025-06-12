//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as ini from 'ini';
import * as path from 'path';
import * as vscode from 'vscode';
import * as config from '../utils/config';
import { realpath } from 'fs-extra';
import { Logger } from '../logging';
import { DisposableContext } from '../utils/disposableContext';
import { MAXSDKConfig } from './sdkConfig';
import { MAXSDK } from './sdk';
import { Mutex } from 'async-mutex';
import {
  directoryExists,
  getAllOpenMojoFiles,
  isMojoFile,
  moveUpUntil,
  readFile,
} from '../utils/files';
import { MAXSDKVersion } from './sdkVersion';
import { findMagicSDKSpec, installMagicSDK } from './magicSdk';
import { Expected, MAXSDKSpec } from './types';
import { Subject } from 'rxjs';

type NotYetSelectedSDK = {
  state: 'not-yet-selected';
};

type SelectedSDK = {
  state: 'selected';
  sdkSpec: Optional<MAXSDKSpec>;
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
export class MAXSDKManager extends DisposableContext {
  public logger: Logger;
  private statusBarItem: vscode.StatusBarItem;
  private _activeSDK: SDKSelection = { state: 'not-yet-selected' };
  private set activeSDK(newSDK: SDKSelection) {
    this._activeSDK = newSDK;
    this.refreshStatusBarItemVisibility();
    this.onActiveSDKChanged.next();
  }
  private get activeSDK() {
    return this._activeSDK;
  }

  private findSDKMutex = new Mutex();
  private isNightly: boolean;
  private extensionContext: vscode.ExtensionContext;
  public onActiveSDKChanged = new Subject<void>();

  constructor(
    logger: Logger,
    isNightly: boolean,
    extensionContext: vscode.ExtensionContext,
  ) {
    super();
    this.logger = logger;
    this.isNightly = isNightly;
    this.extensionContext = extensionContext;
    this.statusBarItem = vscode.window.createStatusBarItem(
      'mojo.selected-sdk',
      vscode.StatusBarAlignment.Right,
      /*priority=*/ 100,
    );
    this.pushSubscription(this.statusBarItem);
    this.statusBarItem.command = 'mojo.sdk.select-default';
    this.statusBarItem.name = 'MAX SDK';
    this.pushSubscription(
      vscode.window.onDidChangeVisibleTextEditors((_editors) => {
        this.refreshStatusBarItemVisibility();
      }),
    );

    this.pushSubscription(
      vscode.commands.registerCommand('mojo.sdk.select-default', async () => {
        const allSDKSpecs = await this.findAllSDKs();
        if (allSDKSpecs.length === 0) {
          vscode.window.showErrorMessage('No MAX SDKs were found.');
          return;
        }
        const sdkNames = allSDKSpecs.map((spec) => spec.version.toString());
        const selectedName = await vscode.window.showQuickPick(sdkNames, {
          ignoreFocusOut: true,
          title: 'Select the default MAX SDK to use',
          placeHolder:
            this.activeSDK.state === 'selected' && this.activeSDK.sdkSpec
              ? `Currently using ${this.activeSDK.sdkSpec.version.toString()}`
              : 'Select an SDK or cancel',
        });
        const selectedSDK = allSDKSpecs.find(
          (spec) => spec.version.toString() === selectedName,
        );
        if (selectedSDK !== undefined) {
          this.extensionContext.globalState.update(
            'mojo.defaultSDKModularHomePath',
            selectedSDK.modularHomePath,
          );
          const finalSDK: SelectedSDK = {
            state: 'selected',
            sdkSpec: selectedSDK,
          };
          await this.createSDKAndShowError(
            finalSDK,
            /*hireRepeatedErrors=*/ false,
            /*allowInstallation=*/ true,
          );
          this.activeSDK = finalSDK;
        }
      }),
    );
    this.pushSubscription(
      vscode.commands.registerCommand('mojo.sdk.reinstall', async () => {
        const result = await installMagicSDK(
          /*withLock=*/ false,
          this.extensionContext,
          this.logger,
          this.isNightly,
          /*reinstall=*/ true,
        );
        if (
          this.activeSDK.state === 'selected' &&
          this.activeSDK.sdkSpec?.kind === 'magic' &&
          this.activeSDK.errorMessage !== undefined &&
          result === 'succeeded'
        ) {
          this.activeSDK = {
            state: 'selected',
            sdkSpec: this.activeSDK.sdkSpec,
          };
        }
      }),
    );
  }

  public async findSDK(hideRepeatedErrors: boolean): Promise<Optional<MAXSDK>> {
    const doWork = async () => {
      if (this.activeSDK.state === 'selected') {
        return this.createSDKAndShowError(
          this.activeSDK,
          hideRepeatedErrors,
          /*allowInstallation=*/ false,
        );
      } else {
        // This is invoked only once per extension activation.
        const sdkSpec = await this.selectSDK();
        const sdkSelection: SDKSelection = { state: 'selected', sdkSpec };
        const sdk = await this.createSDKAndShowError(
          sdkSelection,
          /*hideRepeatedErrors=*/ false,
          /*allowInstallation=*/ true,
        );
        this.activeSDK = sdkSelection;
        return sdk;
      }
    };

    return this.findSDKMutex.runExclusive(() => doWork());
  }

  public async createAdHocSDKAndShowError(
    modularHomePath: string,
  ): Promise<Optional<MAXSDK>> {
    const hideRepeatedErrors = false;
    const allowInstallation = false;

    const devSDKSpec = await this.findDevSDKSpecFromSubPath(modularHomePath);
    if (devSDKSpec !== undefined) {
      return this.createSDKAndShowError(
        { state: 'selected', sdkSpec: devSDKSpec },
        hideRepeatedErrors,
        /*allowInstallation=*/ false,
      );
    }
    if (
      this.activeSDK.state === 'selected' &&
      this.activeSDK.sdkSpec?.modularHomePath === modularHomePath
    ) {
      return this.createSDKAndShowError(
        this.activeSDK,
        hideRepeatedErrors,
        allowInstallation,
      );
    }
    const sdkSpec: MAXSDKSpec = {
      kind: 'custom',
      modularHomePath,
      version: new MAXSDKVersion(
        modularHomePath,
        '0',
        '0',
        '0',
        modularHomePath,
      ),
    };
    return this.createSDKAndShowError(
      { state: 'selected', sdkSpec },
      hideRepeatedErrors,
      allowInstallation,
    );
  }

  private async createSDKAndShowError(
    selectedSDK: SelectedSDK,
    hideRepeatedErrors: boolean,
    allowInstallation: boolean,
  ): Promise<Optional<MAXSDK>> {
    const result = await this.doCreateSDK(selectedSDK, allowInstallation);
    if (result.errorMessage !== undefined) {
      if (
        hideRepeatedErrors &&
        selectedSDK.errorMessage === result.errorMessage
      ) {
        return undefined;
      }
      let errorMessage = result.errorMessage;
      selectedSDK.errorMessage = result.errorMessage;

      if (selectedSDK.sdkSpec?.kind === 'dev') {
        this.showBazelwRunInstallPrompt(
          errorMessage,
          selectedSDK.sdkSpec.modularHomePath,
        );
      } else if (selectedSDK.sdkSpec?.kind === 'magic') {
        errorMessage += '\nPlease reinstall the MAX SDK for VS Code.';
        vscode.window
          .showErrorMessage(errorMessage, 'Reinstall')
          .then(async (value) => {
            if (value === 'Reinstall') {
              vscode.commands.executeCommand('mojo.sdk.reinstall');
            }
          });
      } else if (selectedSDK.sdkSpec?.kind === 'custom') {
        errorMessage += `\nPlease reinstall or rebuild the SDK given by ${selectedSDK.sdkSpec.modularHomePath}.`;
        vscode.window.showErrorMessage(errorMessage);
      }
      this.logger.error(errorMessage);
      return undefined;
    }
    return result.value;
  }

  private refreshStatusBarItemVisibility(): void {
    if (isMojoFile(vscode.window.activeTextEditor?.document.uri)) {
      const activeSDK = this.activeSDK;
      if (activeSDK.state === 'selected' && activeSDK.sdkSpec !== undefined) {
        this.statusBarItem.text = `MAX SDK: ${activeSDK.sdkSpec.version.toTinyString()}`;
        this.statusBarItem.show();
        return;
      }
    }
    this.statusBarItem.hide();
  }

  public showBazelwRunInstallPrompt(
    errorMessage: string,
    modularHomePath: string,
  ): void {
    const action = 'Run ./bazelw run //:install';
    vscode.window.showErrorMessage(errorMessage, action).then((value) => {
      if (value === action) {
        const repo = path.dirname(modularHomePath);
        const terminal =
          vscode.window.activeTerminal ||
          vscode.window.createTerminal({
            name: repo,
          });
        terminal.sendText(`(cd '${repo}' && ./bazelw run //:install)`);
        terminal.show();
      }
    });
  }

  private async doCreateSDK(
    selectedSDK: SelectedSDK,
    allowInstallation: boolean,
  ): Promise<Expected<MAXSDK>> {
    const spec = selectedSDK.sdkSpec;
    if (spec === undefined) {
      return {
        errorMessage: 'The Mojo🔥 development environment was not found.',
      };
    }
    if (selectedSDK.sdkSpec?.kind === 'magic') {
      if (allowInstallation) {
        await installMagicSDK(
          /*withLock=*/ true,
          this.extensionContext,
          this.logger,
          this.isNightly,
        );
      }
    }
    const modularConfigPath = path.join(spec.modularHomePath, 'modular.cfg');
    const modularConfigContents = await readFile(modularConfigPath);
    if (modularConfigContents === undefined) {
      return {
        errorMessage: `The modular config file '${modularConfigPath}' can't be read.`,
      };
    }
    const modularConfig = ini.parse(modularConfigContents);
    this.logger.info(`'${modularConfigPath}' with contents`, modularConfig);
    const mojoConfig = modularConfig['mojo-max'];
    if (!mojoConfig) {
      return {
        errorMessage: `The modular config file '${modularConfigPath}' doesn't have the expected section 'mojo-max'`,
      };
    }
    const sdkConfig = new MAXSDKConfig(
      spec.version,
      spec.modularHomePath,
      mojoConfig,
    );
    if (!sdkConfig) {
      return {
        errorMessage: `Unable to determine the MAX SDK version.`,
      };
    }
    return { value: new MAXSDK(sdkConfig, spec.kind, this.logger) };
  }

  private async selectSDK(): Promise<Optional<MAXSDKSpec>> {
    const allSDKSpecs = await this.findAllSDKs();
    if (allSDKSpecs.length === 0) {
      return undefined;
    }
    if (allSDKSpecs.length === 1) {
      return allSDKSpecs[0];
    }
    const defaultSDKModularHomePath = this.extensionContext.globalState.get<
      Optional<string>
    >('mojo.defaultSDKModularHomePath');
    const selectedDefaultSDKSpec = allSDKSpecs.find(
      (spec) => spec.modularHomePath === defaultSDKModularHomePath,
    )!;
    if (selectedDefaultSDKSpec !== undefined) {
      return selectedDefaultSDKSpec;
    }

    return allSDKSpecs[0];
  }

  private async findAllSDKs(): Promise<MAXSDKSpec[]> {
    // If we're only going to use the release SDK specs, don't bother looking for others.
    if (process.env['MOJO_EXTENSION_FORCE_MAGIC']) {
      const releaseSDKSpecs = await this.findMagicSDKSpecs();
      this.logger.debug(
        'MOJO_EXTENSION_FORCE_MAGIC is set; using release SDK spec(s)',
        releaseSDKSpecs,
      );
      return releaseSDKSpecs;
    }

    // This list has to be returned in a specific order, as the default SDK
    // in new sessions will be the first one from this list.
    const [devSDKSpecs, releaseSDKSpecs, userProvidedSDKSpecs] =
      await Promise.all([
        this.findUserProvidedSDKSpecs(),
        this.findDevSDKSpecs(),
        this.findMagicSDKSpecs(),
      ]);

    return [...devSDKSpecs, ...releaseSDKSpecs, ...userProvidedSDKSpecs];
  }

  private async findUserProvidedSDKSpecs(): Promise<MAXSDKSpec[]> {
    const additionalRoots = config.get<string[]>(
      'SDK.additionalSDKs',
      undefined,
      [],
    );
    const specs = await Promise.all(
      additionalRoots.map(async (modularHomePath) => {
        const modularConfig = path.join(modularHomePath, 'modular.cfg');
        const contents = await readFile(modularConfig);
        if (contents === undefined) {
          this.logger.error(
            `Unable to read ${modularConfig}. Skipping the user-provided SDK ${modularHomePath}`,
          );
          return undefined;
        }
        const spec: MAXSDKSpec = {
          kind: 'custom',
          modularHomePath,
          version: new MAXSDKVersion(
            'MAX SDK',
            '-1',
            '-1',
            '-1',
            modularHomePath,
          ),
        };
        return spec;
      }),
    );
    return specs.filter((spec): spec is MAXSDKSpec => spec !== undefined);
  }

  private async findDevSDKSpecs(): Promise<MAXSDKSpec[]> {
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
    paths: string[],
  ): Promise<MAXSDKSpec[]> {
    const candidateSDKSpecs = (
      await Promise.all(
        paths.map((path) => this.findDevSDKSpecFromSubPath(path)),
      )
    ).filter((spec): spec is MAXSDKSpec => spec !== undefined);
    const uniqueSDKSpecs = new Map<string, MAXSDKSpec>();
    candidateSDKSpecs.forEach((spec) =>
      uniqueSDKSpecs.set(spec.modularHomePath, spec),
    );
    return [...uniqueSDKSpecs.values()];
  }

  private async findDevSDKSpecFromSubPath(
    fsPath: string,
  ): Promise<Optional<MAXSDKSpec>> {
    const repoRoot = await moveUpUntil(fsPath, (p) =>
      directoryExists(path.join(p, '.git')),
    );
    if (!repoRoot) {
      return undefined;
    }
    const bazelPath = path.join(repoRoot, 'MODULE.bazel');
    const bazelContents = await readFile(bazelPath);
    if (!bazelContents) {
      return undefined;
    }
    if (!bazelContents.includes('module(name = "modular-internal")')) {
      return undefined;
    }
    // It is possible to clone the monorepo and run the extension without ever creating .derived.
    if (!directoryExists(path.join(repoRoot, '.derived'))) {
      return undefined;
    }
    const modularHomePath = await realpath(path.join(repoRoot, '.derived'));
    const spec: MAXSDKSpec = {
      kind: 'dev',
      modularHomePath,
      version: new MAXSDKVersion(
        'Modular Repo',
        '0',
        '0',
        '0',
        modularHomePath,
      ),
    };
    return spec;
  }

  private async findMagicSDKSpecs(): Promise<MAXSDKSpec[]> {
    const spec = await findMagicSDKSpec(
      this.extensionContext,
      this.logger,
      this.isNightly,
    );
    return spec ? [spec] : [];
  }
}
