import { MojoSDKVersion } from './sdkVersion';

export type MojoSDKKind = 'dev' | 'magic' | 'custom';

/**
 * A Mojo SDK Spec represents an SDK somewhere in the file system, but it's not
 * guaranteed to exist or even have a valid modular.cfg file.
 */
export type MojoSDKSpec = {
  kind: MojoSDKKind;
  modularHomePath: string;
  section: string;
  version: MojoSDKVersion;
};

export type Expected<T> =
  | {
      errorMessage: string;
    }
  | {
      value: T;
      errorMessage?: undefined;
    };
