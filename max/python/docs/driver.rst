:title: max.driver
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.driver
==========

.. automodule:: max.driver
   :no-members:

.. currentmodule:: max.driver

Devices
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Accelerator
   CPU
   Device
   DeviceEvent
   DeviceSpec
   DeviceStream

Buffers
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Buffer
   DevicePinnedBuffer
   DLPackArray

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   load_max_buffer

Device discovery
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   accelerator_api
   accelerator_architecture_name
   accelerator_count
   devices_exist
   enable_all_peer_access
   load_devices
   scan_available_devices

Virtual devices
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   calculate_virtual_device_count
   calculate_virtual_device_count_from_cli
   get_virtual_device_api
   get_virtual_device_count
   get_virtual_device_target_arch
   is_virtual_device_mode
   set_virtual_device_api
   set_virtual_device_count
   set_virtual_device_target_arch
