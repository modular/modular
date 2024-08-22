//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as path from 'path';

/**
 * This class represents a Mojo SDK version.
 */
export class MojoSDKVersion {
  constructor(
    title: string,
    major: number,
    minor: number,
    patch: number,
    driverPath: string
  ) {
    this.title = title;
    this.minor = minor;
    this.major = major;
    this.patch = patch;
    this.driverPath = driverPath;
  }

  /**
   * Return if this is a dev version.
   */
  isDev(): boolean {
    return this.minor == 0 && this.major == 0 && this.patch == 0;
  }

  /**
   * Convert the version into a human readable string.
   */
  toString(): string {
    // If this is a dev build, format the title differently.
    if (this.isDev()) {
      // We include the path to the modular repo, which is three levels up from
      // the mojo driver path.
      const repo = path.join(path.parse(this.driverPath).dir, '..', '..', '..');
      return `${this.title} (dev) - ${repo}`;
    }

    // Otherwise, just format the version number.
    return `${this.title} (${this.major}.${this.minor}.${this.patch})`;
  }

  title: string;
  minor: number;
  major: number;
  patch: number;
  driverPath: string;
}
