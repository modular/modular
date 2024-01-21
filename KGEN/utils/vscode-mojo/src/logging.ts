//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This file is a modified copy from
// https://github.com/prettier/prettier-vscode/blob/main/src/LoggingService.ts
// which has MIT license.

import * as vscode from 'vscode';
import {window} from "vscode";

type LogLevel = "DEBUG"|"INFO"|"WARN"|"ERROR"|"NONE";

export class LoggingService {
  readonly outputChannel: vscode.OutputChannel;
  private logLevel: LogLevel = "INFO";

  constructor(outputChannelName: string) {
    this.outputChannel = window.createOutputChannel(outputChannelName);
  }

  public setOutputLevel(logLevel: LogLevel) { this.logLevel = logLevel; }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public logDebug(message: string, data?: unknown): void {
    if (this.logLevel === "NONE" || this.logLevel === "INFO" ||
        this.logLevel === "WARN" || this.logLevel === "ERROR") {
      return;
    }
    this.logMessage(message, "DEBUG");
    if (data) {
      this.logObject(data);
    }
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public logInfo(message: string, data?: unknown): void {
    if (this.logLevel === "NONE" || this.logLevel === "WARN" ||
        this.logLevel === "ERROR") {
      return;
    }
    this.logMessage(message, "INFO");
    if (data) {
      this.logObject(data);
    }
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  public logWarning(message: string, data?: unknown): void {
    if (this.logLevel === "NONE" || this.logLevel === "ERROR") {
      return;
    }
    this.logMessage(message, "WARN");
    if (data) {
      this.logObject(data);
    }
  }

  public logError(message: string, error?: unknown) {
    if (this.logLevel === "NONE") {
      return;
    }
    this.logMessage(message, "ERROR");
    if (typeof error === "string") {
      // Errors as a string usually only happen with plugins that don't return
      // the expected error.
      this.outputChannel.appendLine(error);
    } else if (error instanceof Error) {
      if (error?.message) {
        this.logMessage(error.message, "ERROR");
      }
      if (error?.stack) {
        this.outputChannel.appendLine(error.stack);
      }
    } else if (error) {
      this.logObject(error);
    }
  }

  public show() { this.outputChannel.show(); }

  private logObject(data: unknown): void {
    const message = JSON.stringify(data, null, 2);
    this.outputChannel.appendLine(message);
  }

  /**
   * Append messages to the output channel and format it with a title
   *
   * @param message The message to append to the output channel
   */
  private logMessage(message: string, logLevel: LogLevel): void {
    const title = new Date().toLocaleTimeString();
    this.outputChannel.appendLine(`["${logLevel}" - ${title}] ${message}`);
  }

  public dispose(): void { this.outputChannel.dispose(); }
}
