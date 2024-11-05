//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implements a wrapper around @vscode/extension-telemetry to allow us more
// control over the telemetry reporting process.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import BaseTelemetryReporter from '@vscode/extension-telemetry';
import {
  TelemetryEventProperties,
  TelemetryEventMeasurements,
} from '@vscode/extension-telemetry';

export class TelemetryReporter implements vscode.Disposable {
  /**
   * Enables or disables telemetry reporting. If this flag is set to `false`,
   * calls to telemetry reporting methods will become no-ops.
   */
  get enabled() {
    return this.reporter !== undefined && this._enabled;
  }

  set enabled(value) {
    this._enabled = value;
  }

  private _enabled: boolean = true;

  private reporter?: BaseTelemetryReporter;
  private additionalProperties: { [key: string]: string } = {};
  private additionalMeasurements: { [key: string]: number } = {};

  /**
   * Creates a new telemetry reporter. If the connection string is undefined,
   * the reporter will be permanently disabled and will never report telemetry
   * to the remote server.
   */
  constructor(connectionString?: string) {
    if (connectionString) {
      this.reporter = new BaseTelemetryReporter(connectionString);
    }
  }

  public sendTelemetryEvent(
    eventName: string,
    properties?: TelemetryEventProperties,
    measurements?: TelemetryEventMeasurements,
  ): void {
    if (!this.enabled || !this.reporter) {
      return;
    }

    this.reporter.sendTelemetryEvent(
      eventName,
      { ...properties, ...this.additionalProperties },
      { ...measurements, ...this.additionalMeasurements },
    );
  }

  public sendTelemetryErrorEvent(
    eventName: string,
    properties?: TelemetryEventProperties,
    measurements?: TelemetryEventMeasurements,
  ): void {
    if (!this.enabled || !this.reporter) {
      return;
    }

    this.reporter.sendTelemetryErrorEvent(
      eventName,
      { ...properties, ...this.additionalProperties },
      { ...measurements, ...this.additionalMeasurements },
    );
  }

  public dispose() {
    if (this.reporter) {
      this.reporter.dispose();
    }
  }
}
