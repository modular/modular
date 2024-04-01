# TensorFlow: `relu3x100x100-tf`

Generated with the following code:

```python
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((1, 3, 100, 100), dtype="float32", name="input"),
        tf.keras.layers.ReLU(name="relu"),
    ]
)
archive = tf.keras.export.ExportArchive()
archive.track(model)
archive.add_endpoint(
    name="serving_default",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(1, 3, 100, 100), dtype=tf.float32)],
)
archive.write_out("relu3x100x100-tf")
```
