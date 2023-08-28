//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';
import {MOJOSDKConfig} from '../mojoSDK';
import {DisposableContext} from '../utils/disposableContext';

/**
 * This class provides a manager for executing and debugging mojo files.
 */
class ExecutionManager extends DisposableContext {
  _context: MOJOContext|undefined;

  constructor(context: MOJOContext) {
    super();

    this._context = context;
    this.activateRunCommands();
  }

  /**
   * Activate the run commands, used for executing and debugging mojo files.
   */
  activateRunCommands() {
    let execCommands = [
      'mojo.execInTerminal', 'mojo.execInTerminal-icon',
      'mojo.execInDedicatedTerminal'
    ];
    for (const cmd of execCommands) {
      this.pushSubscription(vscode.commands.registerCommand(cmd, async () => {
        await this.executeFileInTerminal({
          newTerminalPerFile : cmd === 'mojo.execInDedicatedTerminal',
        });
      }));
    }

    this.pushSubscription(vscode.commands.registerCommand(
        'mojo.debugInTerminal', async () => { this.debugFileInTerminal(); }));
  }

  /**
   * Execute the current file in a terminal.
   *
   * @param options Options to consider when executing the file.
   */
  async executeFileInTerminal(options: {newTerminalPerFile: boolean}) {
    let doc = await this.getFileToExecute();
    if (!doc)
      return;

    // Find the config for processing this file.
    let config = await this._context?.getSDK().resolveConfig(
        vscode.workspace.getWorkspaceFolder(doc.uri));
    if (!config)
      return;

    // Execute the file.
    let terminal =
        this.getTerminalForFile(doc, config, options.newTerminalPerFile);
    terminal.sendText(config.mojoDriverPath + ' ' + doc.fileName);

    // Focus on the terminal if the user has configured it to do so.
    if (this.shouldTerminalFocusOnStart(doc.uri))
      vscode.commands.executeCommand('workbench.action.terminal.focus');
  }

  /**
   * Debug the current file in a terminal.
   */
  async debugFileInTerminal() {
    let doc = await this.getFileToExecute();
    if (!doc)
      return;

    // Find the config for processing this file.
    let config = await this._context?.getSDK().resolveConfig(
        vscode.workspace.getWorkspaceFolder(doc.uri));
    if (config === undefined)
      return;

    // Pull in the additional visualizers within the lldb-visualizers dir.
    let visualizersDir = config.mojoLLDBVisualizersPath;
    let visualizers = await vscode.workspace.fs.readDirectory(
        vscode.Uri.file(visualizersDir));
    let visualizerCommands = visualizers.map(
        ([ name, _type ]) => `command script import ${visualizersDir}/${name}`);

    let debugConfig: vscode.DebugConfiguration = {
      type : "mojo-lldb",
      name : "Mojo",
      request : "launch",
      program : config.mojoDriverPath,
      args :
          [ "run", "--no-optimization", "--debug-level", "full", doc.fileName ],
      env : {
        "MODULAR_HOME" : config.modularHomePath,
      },
      initCommands : visualizerCommands,
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
   * Get the currently active file to execute.
   */
  async getFileToExecute(): Promise<vscode.TextDocument|undefined> {
    let doc = vscode.window.activeTextEditor?.document;
    if (!doc)
      return undefined;
    if (doc.isDirty && !await doc.save())
      return undefined;
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
