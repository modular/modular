import { MojoSDKVersion } from './sdkVersion';

export type MojoSDKSpec = {
  kind: 'modular-cli' | 'dev' | 'magic' | 'custom';
  modularHomePath: string;
  section: string;
  version: MojoSDKVersion;
};
