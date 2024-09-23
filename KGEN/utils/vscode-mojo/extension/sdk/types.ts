import { MojoSDKVersion } from './sdkVersion';

/**
 * A Mojo SDK Spec represents an SDK somewhere in the file system, but it's not
 * guaranteed to exist or even have a valid modular.cfg file.
 */
export type MojoSDKSpec = {
  kind: 'modular-cli' | 'dev' | 'magic' | 'custom';
  modularHomePath: string;
  section: string;
  version: MojoSDKVersion;
};
