:title: driver
:type: module
:lang: python
:source: max/driver/

Exposes APIs for interacting with hardware, such as allocating tensors on a GPU
and moving tensors between the CPU and GPU. It provides interfaces for memory
management, device properties, and hardware monitoring. Through these APIs, you
can control data placement, track resource utilization, and configure device
settings for optimal performance.

For example, you can use the following code to use an accelerator if one is
available, otherwise use the CPU:

.. code-block:: python

   from max import driver

   device = driver.CPU() if driver.accelerator_count() == 0 else driver.Accelerator()
   print(f"Using {device} device")

.. automodule:: max.driver
   :imported-members:
   :members:
   :undoc-members:

.. autofunction:: max.driver.accelerator_count
