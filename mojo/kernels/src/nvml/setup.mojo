# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host import device_count

from .nvml import Device


def _set_persistence_mode(device: Device):
    try:
        device.set_persistence_mode(enabled=True)
    except:
        print("unable to set the gpu persistence for", device)


def _disable_gpu_turbo(device: Device):
    try:
        device.set_gpu_turbo(enabled=False)
    except:
        print("unable to set the gpu turbo for", device)


def lock_gpus(continue_on_error: Bool = True):
    for device_id in range(device_count()):
        try:
            var device = Device(device_id)
            _set_persistence_mode(device)
            _disable_gpu_turbo(device)
            device.set_max_gpu_clocks()
        except e:
            print(e)
            if not continue_on_error:
                return
