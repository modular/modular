import { DisposableContext } from '../utils/disposableContext';
import * as vscode from 'vscode';
import { MessageSignature, CancellationToken } from 'vscode-languageclient';
import { createWriteStream, WriteStream } from 'fs';

export class LSPRecorder extends DisposableContext {
  private output: WriteStream;

  constructor(outPath: string) {
    super();

    this.output = createWriteStream(outPath);

    this.pushSubscription(
      new vscode.Disposable(() => {
        this.output.close();
      }),
    );
  }

  // Follows GeneralMiddleware implementation from vscode-languageclient.
  public sendRequest<P, R>(
    type: string | MessageSignature,
    param: P | undefined,
    token: CancellationToken | undefined,
    next: (
      type: string | MessageSignature,
      param?: P,
      token?: CancellationToken,
    ) => Promise<R>,
  ): Promise<R> {
    const message = {
      type: 'request',
      method: type,
      param: param,
    };

    this.output.write(JSON.stringify(message));
    this.output.write('\n');
    return next(type, param, token);
  }

  public sendNotification<P>(
    type: string | MessageSignature,
    next: (type: string | MessageSignature, params?: P) => Promise<void>,
    param: P,
  ): Promise<void> {
    const message = {
      type: 'notification',
      method: type,
      param: param,
    };

    this.output.write(JSON.stringify(message));
    this.output.write('\n');
    return next(type, param);
  }
}
