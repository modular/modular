//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { quote, parse } from 'shell-quote';
import * as vscode from 'vscode';
import { DisposableContext } from '../utils/disposableContext';
import path = require('path');
import * as config from '../utils/config';
import { MAXSDK } from '../sdk/sdk';
import { MAXSDKManager } from '../sdk/sdkManager';
import { MojoDebugConfiguration } from '../debug/debug';
import * as md5 from 'md5';

type FileArgs = {
  runArgs: string[];
  buildArgs: string[];
};

/**
 * This class provides a manager for executing and debugging mojo files.
 */
class ExecutionManager extends DisposableContext {
  readonly sdkManager: MAXSDKManager;
  private context: vscode.ExtensionContext;

  constructor(sdkManager: MAXSDKManager, context: vscode.ExtensionContext) {
    super();

    this.sdkManager = sdkManager;
    this.context = context;
    this.activateRunCommands();
  }

  private getFileArgsKey(path: string): string {
    return `file.args.${path}`;
  }

  private getFileArgs(path: string): FileArgs {
    return this.context.globalState.get<FileArgs>(this.getFileArgsKey(path), {
      runArgs: [],
      buildArgs: [],
    });
  }

  private getBuildArgs(path: string): string[] {
    return this.getFileArgs(path).buildArgs;
  }

  private async setBuildArgs(path: string, args: string): Promise<void> {
    const fileArgs = this.getFileArgs(path);
    fileArgs.buildArgs = parse(args).filter(
      (x): x is string => typeof x === 'string',
    );
    return this.context.globalState.update(this.getFileArgsKey(path), fileArgs);
  }

  private getRunArgs(path: string): string[] {
    return this.getFileArgs(path).runArgs;
  }

  private async setRunArgs(path: string, args: string): Promise<void> {
    const fileArgs = this.getFileArgs(path);
    fileArgs.runArgs = parse(args).filter(
      (x): x is string => typeof x === 'string',
    );
    return this.context.globalState.update(this.getFileArgsKey(path), fileArgs);
  }

  /**
   * Activate the run commands, used for executing and debugging mojo files.
   */
  activateRunCommands() {
    const cmd = 'mojo.file.run';
    this.pushSubscription(
      vscode.commands.registerCommand(cmd, (file?: vscode.Uri) => {
        this.executeFileInTerminal(file);
        return true;
      }),
    );

    for (const cmd of ['mojo.file.debug', 'mojo.file.debug-in-terminal']) {
      this.pushSubscription(
        vscode.commands.registerCommand(cmd, (file: vscode.Uri) => {
          this.debugFile(
            file,
            /*runInTerminal=*/ cmd === 'mojo.file.debug-in-terminal',
          );
          return true;
        }),
      );
    }
    this.pushSubscription(
      vscode.commands.registerCommand(
        'mojo.file.set-args',
        async (file: vscode.Uri) => {
          const setBuildArgs = 'Set Build Arguments';
          const setRunArgs = 'Set Run Arguments';
          const option = await vscode.window.showQuickPick(
            [setBuildArgs, setRunArgs],
            {
              title: 'Select the arguments you want to configure',
              placeHolder:
                'This will affect `Run Mojo File`, `Debug Mojo File` and similar actions.',
            },
          );

          if (option === setBuildArgs) {
            const buildArgs = quote(this.getBuildArgs(file.fsPath));

            const newValue = await vscode.window.showInputBox({
              placeHolder: 'Enter the arguments as if within a shell.',
              title: 'Enter the build arguments for the compiler',
              value: buildArgs.length === 0 ? undefined : buildArgs,
            });
            if (newValue !== undefined) {
              await this.setBuildArgs(file.fsPath, newValue);
            }
          } else if (option === setRunArgs) {
            const runArgs = quote(this.getRunArgs(file.fsPath));

            const newValue = await vscode.window.showInputBox({
              placeHolder: 'Enter the arguments as if within a shell.',
              title: 'Enter the run arguments for the final executable',
              value: runArgs.length === 0 ? undefined : runArgs,
            });
            if (newValue !== undefined) {
              await this.setRunArgs(file.fsPath, newValue);
            }
          }
          return true;
        },
      ),
    );
  }

  /**
   * Execute the current file in a terminal.
   *
   * @param options Options to consider when executing the file.
   */
  async executeFileInTerminal(file: Optional<vscode.Uri>) {
    let doc = await this.getDocumentToExecute(file);

    if (!doc) {
      return;
    }

    // Find the config for processing this file.
    let sdk = await this.sdkManager.findSDK(/*hideRepeatedErrors=*/ false);

    if (!sdk) {
      return;
    }

    // Execute the file.
    let terminal = this.getTerminalForFile(doc, sdk);
    terminal.show();
    terminal.sendText(
      quote([
        sdk.config.mojoDriverPath,
        'run',
        ...this.getBuildArgs(doc.fileName),
        doc.fileName,
        ...this.getRunArgs(doc.fileName),
      ]),
    );

    if (this.shouldTerminalFocusOnStart(doc.uri)) {
      vscode.commands.executeCommand('workbench.action.terminal.focus');

      // Sometimes VSCode will focus on the terminal as a side-effect of `terminal.show()`,
      // in which case we need to indicate it to switch back the focus to the previous
      // focus group.
    } else {
      vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
    }
  }

  /**
   * Debug the current file.
   *
   * @param runInTerminal If true, then a target is launched in a new
   *     terminal, and therefore its stdin and stdout are not managed by the
   *     Debug Console.
   */
  async debugFile(file: Optional<vscode.Uri>, runInTerminal: boolean) {
    let doc = await this.getDocumentToExecute(file);

    if (!doc) {
      return;
    }

    let debugConfig: MojoDebugConfiguration = {
      type: 'mojo-lldb',
      name: 'Mojo',
      request: 'launch',
      mojoFile: doc.fileName,
      runInTerminal: runInTerminal,
      buildArgs: this.getBuildArgs(doc.fileName),
      args: this.getRunArgs(doc.fileName),
    };
    await vscode.debug.startDebugging(
      vscode.workspace.getWorkspaceFolder(doc.uri),
      debugConfig as vscode.DebugConfiguration,
    );
  }

  /**
   * Get a terminal to use for the given file.
   */
  getTerminalForFile(doc: vscode.TextDocument, sdk: MAXSDK): vscode.Terminal {
    const fullId = `${doc.fileName} · ${sdk.config.modularHomePath}`;
    // We have to keep the full terminal name short so that VS Code renders it nicely,
    // and we have to keep it unique among other files.
    let terminalName = `Mojo: ${path.basename(doc.fileName)} · ${md5(fullId).substring(0, 5)}`;

    // Look for an existing terminal.
    let terminal = vscode.window.terminals.find((t) => t.name === terminalName);

    if (terminal) {
      return terminal;
    }

    // Build a new terminal.
    return vscode.window.createTerminal({
      name: terminalName,
      env: sdk.getProcessEnv(),
      hideFromUser: true,
    });
  }

  /**
   * Get the vscode.Document to execute, ensuring that it's saved if pending
   * changes exist.
   *
   * This method show a pop up in case of errors.
   *
   * @param file If provided, the document will point to this file, otherwise,
   *     it will point to the currently active document.
   */
  async getDocumentToExecute(
    file?: vscode.Uri,
  ): Promise<Optional<vscode.TextDocument>> {
    let doc =
      file === undefined
        ? vscode.window.activeTextEditor?.document
        : await vscode.workspace.openTextDocument(file);
    if (!doc) {
      vscode.window.showErrorMessage(
        `Couldn't access the file '${file}' for execution.`,
      );
      return undefined;
    }
    if (doc.isDirty && !(await doc.save())) {
      vscode.window.showErrorMessage(
        `Couldn't save file '${file}' before execution.`,
      );
      return undefined;
    }
    return doc;
  }

  /**
   * Returns true if the terminal should be focused on start.
   */
  private shouldTerminalFocusOnStart(uri: vscode.Uri): boolean {
    return config.get<boolean>(
      'run.focusOnTerminalAfterLaunch',
      vscode.workspace.getWorkspaceFolder(uri),
      false,
    );
  }
}

/**
 * Activate the run commands, used for executing and debugging mojo files.
 *
 * @returns A disposable connected to the lifetime of the registered run
 *     commands.
 */
export function activateRunCommands(
  sdkManager: MAXSDKManager,
  context: vscode.ExtensionContext,
): vscode.Disposable {
  return new ExecutionManager(sdkManager, context);
}
