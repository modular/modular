//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from 'path';
/**
 * This class represents a MAX SDK version.
 */
export class MAXSDKVersion {
  constructor(
    title: string,
    major: string,
    minor: string,
    patch: string,
    modularHomePath: string,
  ) {
    this.title = title;
    this.minor = minor;
    this.major = major;
    this.patch = patch;
    this.modularHomePath = modularHomePath;
  }

  /**
   * Return if this is a dev version.
   */
  isDev(): boolean {
    return this.minor === '0' && this.major === '0' && this.patch === '0';
  }

  /**
   * Return if this is a user-provided spec
   */
  isUserProvided(): boolean {
    return this.minor === '-1' && this.major === '-1' && this.patch === '-1';
  }

  /**
   * Convert the version into a human readable string.
   */
  toString(): string {
    // If this is a dev build, format the title differently.
    if (this.isDev()) {
      // We include the path to the modular repo, which is three levels up from
      // the mojo driver path.
      return `${this.title} (dev) - ${this.modularHomePath}`;
    }
    if (this.isUserProvided()) {
      return `${this.title} (user-provided) - ${this.modularHomePath}`;
    }

    // Otherwise, just format the version number.
    return `${this.title} - ${this.major}.${this.minor}.${this.patch}`;
  }

  /**
   * @returns a tiny representation of this version for small displays.
   */
  toTinyString(): string {
    if (this.isDev() || this.isUserProvided()) {
      return `${path.dirname(this.modularHomePath)}`;
    }
    return `${this.title}`;
  }

  title: string;
  minor: string;
  major: string;
  patch: string;
  modularHomePath: string;
}
