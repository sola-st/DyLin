import tensorflow as tf

tf.get_logger().setLevel('INFO')


def basic():
    x = tf.constant([float('nan')])  # DyLin warn
    x = tf.constant([float('inf')])  # DyLin warn
    x = tf.constant([float('-inf')])  # DyLin warn
    x2 = tf.constant([[float('-inf'), float('nan')], [float('nan'), float('nan')]])  # DyLin warn
    x3 = tf.constant(  # DyLin warn
        [[[float(2), float(0)], [float('-inf'), float(2)]], [[float(2), float(2)], [float(2), float(2)]]]
    )

    # could be improved by storing tensors that are already checked
    tf.argmax(x)  # DyLin warn
    tf.argmax(x2)  # DyLin warn

    y_ = tf.constant([float(0)])
    y_conv = tf.constant([float(0)])
    y_ * tf.math.log(y_conv)  # DyLin warn


basic()


def advanced():
    class Custom_CE_Loss(tf.keras.losses.Loss):
        def __init__(self):
            super().__init__()

        def call(self, y_true, y_pred):
            with tf.GradientTape() as tape:
                log_y_pred = tf.math.log(y_pred)  # DyLin warn
                elements = -tf.math.multiply(x=log_y_pred, y=y_true)  # DyLin warn
                _sum = tf.reduce_sum(elements, axis=1)  # DyLin warn
                res = tf.reduce_mean(_sum)  # DyLin warn
            return self.correct_loss(y_true, y_pred)

        def correct_loss(self, y_true, y_pred):
            log_y_pred = tf.math.log(y_pred)  # DyLin warn
            # NaN has been removed, inf still exists
            elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)  # DyLin warn
            # No longer any issues
            return tf.reduce_mean(tf.reduce_sum(elements, axis=1))

    y_true = tf.constant(tf.keras.utils.to_categorical([4, 1]))
    y_pred = tf.constant([[0, 0.7, 0, 0, 0.3], [0, 0.6, 0.3, 0, 0.1]])
    Custom_CE_Loss()(y_true, y_pred)


advanced()
