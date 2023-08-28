//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from './mojoContext';

/**
 * This class defines a factory used to create the debug adaptor used for
 * debugging mojo, which is dependent on the SDK installation.
 */
class MojoDebugAdapterDescriptorFactory implements
    vscode.DebugAdapterDescriptorFactory {
  _context: MOJOContext|undefined;
  public static DEBUG_TYPE: string = "mojo-lldb";

  constructor(context: MOJOContext) { this._context = context; }

  async createDebugAdapterDescriptor(session: vscode.DebugSession,
                                     executable: vscode.DebugAdapterExecutable|
                                     undefined):
      Promise<vscode.DebugAdapterDescriptor|null> {
    let config =
        await this._context?.getSDK().resolveConfig(session.workspaceFolder);
    if (!config)
      return null;
    return new vscode.DebugAdapterExecutable(config.mojoLLDBVSCodePath, []);
  }
}

/**
 * Registers the necessary constructs to support mojo debugging.
 */
export function registerDebugging(context: MOJOContext): vscode.Disposable {
  return vscode.debug.registerDebugAdapterDescriptorFactory(
      MojoDebugAdapterDescriptorFactory.DEBUG_TYPE,
      new MojoDebugAdapterDescriptorFactory(context));
}
