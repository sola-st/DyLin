import tensorflow as tf

tf.get_logger().setLevel('INFO')

d = {"Tensorflow NaN / inf Analysis": "TensorflowNonFinitesAnalysis"}


def basic():
    f'START;'
    x = tf.constant([float('nan')])
    f'END;'
    f'START;'
    x = tf.constant([float('inf')])
    f'END;'
    f'START;'
    x = tf.constant([float('-inf')])
    f'END;'
    f'START;'
    x2 = tf.constant([[float('-inf'), float('nan')],
                     [float('nan'), float('nan')]])
    f'END;'
    f'START;'
    x3 = tf.constant([[[float(2), float(0)], [float('-inf'), float(2)]],
                     [[float(2), float(2)], [float(2), float(2)]]])
    f'END;'

    # should not throw error because x, x2 have already been found
    tf.argmax(x)
    tf.argmax(x2)

    y_ = tf.constant([float(0)])
    y_conv = tf.constant([float(0)])
    f'START;'
    y_*tf.math.log(y_conv)
    f'END; y_*tf.log(y_conv)'


basic()


def advanced():
    class Custom_CE_Loss(tf.keras.losses.Loss):
        def __init__(self):
            super().__init__()

        def call(self, y_true, y_pred):
            with tf.GradientTape() as tape:
                f'START;'
                log_y_pred = tf.math.log(y_pred)
                f'END tf.math.log is inf'
                f'START;'
                elements = -tf.math.multiply(x=log_y_pred, y=y_true)
                f'END tf.math.multiply is NaN'
                f'START;'
                _sum = tf.reduce_sum(elements, axis=1)
                f'END; reduce sum'
                f'START;'
                res = tf.reduce_mean(_sum)
                f'END; reduce sum contains NaN'
            return self.correct_loss(y_true, y_pred)

        def correct_loss(self, y_true, y_pred):
            f'START;'
            log_y_pred = tf.math.log(y_pred)
            f'END;'
            # NaN has been removed, no issues after this point
            # TODO: maybe remove issue if problematic tensor only occurs once
            elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
            return tf.reduce_mean(tf.reduce_sum(elements, axis=1))

    y_true = tf.constant(tf.keras.utils.to_categorical([4, 1]))
    y_pred = tf.constant([[0, .7, 0, 0,  .3], [0, .6, .3, 0,  .1]])
    Custom_CE_Loss()(y_true, y_pred)

advanced()
