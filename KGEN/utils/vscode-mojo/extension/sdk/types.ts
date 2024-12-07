import { MAXSDKVersion } from './sdkVersion';

export type MAXSDKKind = 'dev' | 'magic' | 'custom';

/**
 * A MAX SDK Spec represents an SDK somewhere in the file system, but it's not
 * guaranteed to exist or even have a valid modular.cfg file.
 */
export type MAXSDKSpec = {
  kind: MAXSDKKind;
  modularHomePath: string;
  version: MAXSDKVersion;
};

export type Expected<T> =
  | {
      errorMessage: string;
    }
  | {
      value: T;
      errorMessage?: undefined;
    };
