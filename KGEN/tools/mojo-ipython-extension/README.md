## IPython `%%mojo` Extension

This extension enables `%%mojo` as a IPython Magic Cell.

### Build

```sh
bb //KGEN/tools/mojo-ipython-extension:IPythonExtension
```

### Loading the IPythonExtension

To auto-load in IPython link the provided `ipython_config.py`

```sh
ln -s \
  ${MODULAR_PATH}/KGEN/tools/mojo-ipython-extension/test/ipython_config.py \
  ${HOME}/.ipython/profile_default/ipython_config.py
```
