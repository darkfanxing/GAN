import tensorflow as tf

def check_and_use_gpus(memory_limit: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )

            tf.config.set_visible_devices(gpus[0], "GPU")

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Find {len(gpus)} Physical GPUs, {len(logical_gpus)}, Logical GPU")
        except RuntimeError as error:
            raise(error)                                                            