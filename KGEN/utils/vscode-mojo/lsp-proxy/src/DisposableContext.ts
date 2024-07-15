//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { Disposable } from './types';

export class DisposableCallback implements Disposable {
  private callback: () => void;

  constructor(callback: () => void) {
    this.callback = callback;
  }

  dispose(): void {
    this.callback();
  }
}

/**
 * This class provides a simple wrapper around `Disposable` that allows for
 * registering additional disposables.
 *
 * Note: We can't use vscode.Disposable because the proxy can't depend on the
 * VSCode API.
 */
export class DisposableContext implements Disposable {
  private _disposables: Disposable[] = [];

  constructor() {}

  public dispose() {
    for (const disposable of this._disposables) {
      disposable.dispose();
    }
    this._disposables = [];
  }

  /**
   * Push an additional disposable to the context.
   *
   * @param disposable The disposable to register.
   */
  public pushSubscription(disposable: Disposable) {
    this._disposables.push(disposable);
  }
}
