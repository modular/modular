//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This file is a modified copy from
// https://github.com/prettier/prettier-vscode/blob/main/src/logger.ts
// which has MIT license.

import * as vscode from 'vscode';
import { window } from 'vscode';

import { DisposableContext } from './utils/disposableContext';

export enum LogLevel {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  None = 5,
}

const logLevelToString = (level: LogLevel) => {
  switch (level) {
    case LogLevel.Trace:
      return 'TRACE';
    case LogLevel.Debug:
      return 'DEBUG';
    case LogLevel.Info:
      return 'INFO';
    case LogLevel.Warn:
      return 'WARN';
    case LogLevel.Error:
      return 'ERROR';
    case LogLevel.None:
      return 'NONE';
  }
};

export class LogChannel {
  readonly outputChannel: vscode.OutputChannel;
  private logLevel: LogLevel = LogLevel.Info;
  public logCallback?: (level: string, message: string) => void;

  constructor(outputChannelName: string) {
    this.outputChannel = window.createOutputChannel(outputChannelName);
  }

  public setOutputLevel(logLevel: LogLevel) {
    this.logLevel = logLevel;
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public trace(message: string, data?: unknown): void {
    this.log(LogLevel.Trace, message);
    if (data) {
      this.log(LogLevel.Trace, data);
    }
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public debug(message: string, data?: unknown): void {
    this.log(LogLevel.Debug, message);
    if (data) {
      this.log(LogLevel.Debug, data);
    }
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public info(message: string, data?: unknown): void {
    this.log(LogLevel.Info, message);
    if (data) {
      this.log(LogLevel.Info, data);
    }
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public warn(message: string, data?: unknown): void {
    this.log(LogLevel.Warn, message);
    if (data) {
      this.log(LogLevel.Warn, data);
    }
  }

  public error(message: string, error?: unknown) {
    this.log(LogLevel.Error, message);
    if (typeof error === 'string') {
      // Errors as a string usually only happen with plugins that don't return
      // the expected error.
      this.log(LogLevel.Error, error);
    } else if (error instanceof Error) {
      if (error?.message) {
        this.log(LogLevel.Error, error.message);
      }
      if (error?.stack) {
        this.log(LogLevel.Error, error.stack);
      }
    } else if (error) {
      this.log(LogLevel.Error, error);
    }
  }

  public show() {
    this.outputChannel.show();
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  private log(logLevel: LogLevel, message: unknown): void {
    if (this.logLevel > logLevel) {
      return;
    }

    if (typeof message !== 'string') {
      message = JSON.stringify(message, null, 2);
    }

    const title = new Date().toLocaleTimeString();
    this.outputChannel.appendLine(
      `["${logLevelToString(logLevel)}" - ${title}] ${message}`,
    );

    if (this.logCallback) {
      // tsc doesn't understand that `message` is guaranteed to be a string at this point.
      this.logCallback(logLevelToString(logLevel), message as string);
    }
  }

  public dispose(): void {
    this.outputChannel.dispose();
  }
}

export class Logger extends DisposableContext {
  public main: LogChannel;
  public lsp: LogChannel;

  constructor(isNightly: boolean) {
    super();

    const suffix = isNightly ? ' (nightly)' : '';
    this.main = new LogChannel('Mojo' + suffix);
    this.lsp = new LogChannel('Mojo Language Server' + suffix);

    if (isNightly) {
      this.main.setOutputLevel(LogLevel.Debug);
      this.lsp.setOutputLevel(LogLevel.Debug);
    }

    this.pushSubscription(this.main);
    this.pushSubscription(this.lsp);
  }

  /**
   * Logs a TRACE message to the main log channel.
   */
  public trace(message: string, data?: unknown) {
    this.main.trace(message, data);
  }

  /**
   * Logs a DEBUG message to the main log channel.
   */
  public debug(message: string, data?: unknown) {
    this.main.debug(message, data);
  }

  /**
   * Logs an INFO message to the main log channel.
   */
  public info(message: string, data?: unknown) {
    this.main.info(message, data);
  }

  /**
   * Logs a WARN message to the main log channel.
   */
  public warn(message: string, data?: unknown) {
    this.main.warn(message, data);
  }

  /**
   * Logs an ERROR message to the main log channel.
   */
  public error(message: string, data?: unknown) {
    this.main.error(message, data);
  }
}
