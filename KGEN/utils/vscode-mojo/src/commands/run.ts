//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as shellescape from 'shell-escape';
import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';
import {MOJOSDKConfig} from '../mojoSDK';
import {DisposableContext} from '../utils/disposableContext';

/**
 * This class provides a manager for executing and debugging mojo files.
 */
class ExecutionManager extends DisposableContext {
  readonly context: MOJOContext;

  constructor(context: MOJOContext) {
    super();

    this.context = context;
    this.activateRunCommands();
  }

  /**
   * Activate the run commands, used for executing and debugging mojo files.
   */
  activateRunCommands() {
    for (const cmd of ['mojo.execFileInTerminal',
                       'mojo.execFileInDedicatedTerminal']) {
      this.pushSubscription(
          vscode.commands.registerCommand(cmd, async (file?: vscode.Uri) => {
            await this.executeFileInTerminal(
                file,
                /*newTerminalPerFile=*/ cmd ===
                    'mojo.execFileInDedicatedTerminal',
            );
          }));
    }

    for (const cmd of ['mojo.debugFile', 'mojo.debugFileInTerminal']) {
      this.pushSubscription(
          vscode.commands.registerCommand(cmd, async (file?: vscode.Uri) => {
            await this.debugFile(file, /*runInTerminal=*/ cmd ===
                                           'mojo.debugFileInTerminal');
          }));
    }
  }

  /**
   * Execute the current file in a terminal.
   *
   * @param options Options to consider when executing the file.
   */
  async executeFileInTerminal(file: vscode.Uri|undefined,
                              newTerminalPerFile: boolean) {
    let doc = await this.getDocumentToExecute(file);
    if (!doc)
      return;

    // Find the config for processing this file.
    let config = await this.context.sdk.resolveConfig(
        vscode.workspace.getWorkspaceFolder(doc.uri));
    if (!config)
      return;

    // Execute the file.
    let terminal = this.getTerminalForFile(doc, config, newTerminalPerFile);
    terminal.show();
    terminal.sendText(shellescape([ config.mojoDriverPath, doc.fileName ]));

    // Focus on the terminal if the user has configured it to do so.
    if (this.shouldTerminalFocusOnStart(doc.uri))
      vscode.commands.executeCommand('workbench.action.terminal.focus');
  }

  /**
   * Debug the current file.
   *
   * @param runInTerminal If true, then a target is launched in a new
   *     terminal, and therefore its stdin and stdout are not managed by the
   *     Debug Console.
   */
  async debugFile(file: vscode.Uri|undefined, runInTerminal: boolean) {
    let doc = await this.getDocumentToExecute(file);
    if (!doc)
      return;

    let debugConfig: vscode.DebugConfiguration = {
      type : "mojo-lldb",
      name : "Mojo",
      request : "launch",
      mojoFile : doc.fileName,
      runInTerminal : runInTerminal,
    };
    await vscode.debug.startDebugging(
        vscode.workspace.getWorkspaceFolder(doc.uri), debugConfig);
  }

  /**
   * Get a terminal to use for the given file.
   */
  getTerminalForFile(doc: vscode.TextDocument, config: MOJOSDKConfig,
                     newTerminalPerFile: boolean): vscode.Terminal {
    let terminalName = "Mojo";
    if (newTerminalPerFile)
      terminalName += `: ${doc.fileName}`;

    // Look for an existing terminal.
    let terminal = vscode.window.terminals.find((t) => t.name === terminalName);
    if (terminal)
      return terminal;

    // Build a new terminal.
    let env = process.env;
    env['MODULAR_HOME'] = config.modularHomePath;
    return vscode.window.createTerminal({name : terminalName, env : env});
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
  async getDocumentToExecute(file?: vscode.Uri):
      Promise<vscode.TextDocument|undefined> {
    let doc = file === undefined
                  ? vscode.window.activeTextEditor?.document
                  : await vscode.workspace.openTextDocument(file);
    if (!doc) {
      vscode.window.showErrorMessage(
          `Couldn't access the file '${file}' for execution.`);
      return undefined;
    }
    if (doc.isDirty && !await doc.save()) {
      vscode.window.showErrorMessage(
          `Couldn't save file '${file}' before execution.`);
      return undefined;
    }
    return doc;
  }

  /**
   * Returns true if the terminal should be focused on start.
   */
  private shouldTerminalFocusOnStart(uri: vscode.Uri): boolean {
    return vscode.workspace
        .getConfiguration('terminal', vscode.workspace.getWorkspaceFolder(uri))
        .get<boolean>("focusAfterLaunch", false);
  }
}

/**
 * Activate the run commands, used for executing and debugging mojo files.
 *
 * @param context The MOJO context to use.
 * @returns A disposable connected to the lifetime of the registered run
 *     commands.
 */
export function activateRunCommands(context: MOJOContext): vscode.Disposable {
  return new ExecutionManager(context);
}
