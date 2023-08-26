//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as chokidar from 'chokidar';
import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';

import * as config from './config';

/**
 *  Prompt the user to see if we should restart the server.
 */
async function promptRestart(settingName: string, promptMessage: string) {
  switch (config.get<string>(settingName, /*workspaceFolder=*/ undefined)) {
  case 'restart':
    vscode.commands.executeCommand('mojo.restart');
    break;
  case 'ignore':
    break;
  case 'prompt':
  default:
    switch (await vscode.window.showInformationMessage(
        promptMessage, 'Yes', 'Yes, always', 'No, never')) {
    case 'Yes':
      vscode.commands.executeCommand('mojo.restart');
      break;
    case 'Yes, always':
      vscode.commands.executeCommand('mojo.restart');
      config.update<string>(settingName, 'restart',
                            vscode.ConfigurationTarget.Global);
      break;
    case 'No, never':
      config.update<string>(settingName, 'ignore',
                            vscode.ConfigurationTarget.Global);
      break;
    default:
      break;
    }
    break;
  }
}

/**
 *  Activate watchers that track configuration changes for the given workspace
 *  folder, or null if the workspace is top-level.
 */
export async function activate(mojoContext: MOJOContext,
                               workspaceFolder: vscode.WorkspaceFolder,
                               settings: string[], paths: string[]) {
  // When a configuration change happens, check to see if we should restart.
  mojoContext.pushSubscription(vscode.workspace.onDidChangeConfiguration(event => {
    for (const setting of settings) {
      const expandedSetting = `mojo.${setting}`;
      if (event.affectsConfiguration(expandedSetting, workspaceFolder)) {
        promptRestart(
            'onSettingsChanged',
            `setting '${
                expandedSetting}' has changed. Do you want to reload the server?`);
      }
    }
  }));

  // Setup watchers for the provided paths.
  const fileWatcherConfig = {
    disableGlobbing : true,
    followSymlinks : true,
    ignoreInitial : true,
    awaitWriteFinish : true,
  };
  for (const serverPath of paths) {
    if (serverPath === '') {
      return;
    }

    // If the path actually exists, track it in case it changes.
    const fileWatcher = chokidar.watch(serverPath, fileWatcherConfig);
    fileWatcher.on('all', (event, _filename, _details) => {
      if (event != 'unlink') {
        promptRestart(
            'onSettingsChanged',
            'mojo language server file has changed. Do you want to reload the server?');
      }
    });
    mojoContext.pushSubscription(
        new vscode.Disposable(() => { fileWatcher.close(); }));
  }
}
