import larq as lq
import larq_compute_engine as lce
import tensorflow as tf
import numpy as np
from larq import quantizers
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""


class DPReLU(tf.keras.layers.Layer):
    def __init__(self):
        super(DPReLU, self).__init__()

    def build(self, input_shape):
        self.bias_alpha = self.add_weight("alpha", shape=[1, 1, 1, input_shape[-1]], initializer='zeros',
                                          trainable=True)
        self.bias_beta = self.add_weight("beta", shape=[1, 1, 1, input_shape[-1]], initializer='zeros', trainable=True)
        self.pos_slope = self.add_weight("pos_slope", shape=[1, 1, 1, input_shape[-1]], trainable=True,
                                         initializer=tf.keras.initializers.Constant(1.))
        self.neg_slope = self.add_weight("neg_slope", shape=[1, 1, 1, input_shape[-1]], trainable=True,
                                         initializer=tf.keras.initializers.Constant(0.25))

    def call(self, x):
        x = x - self.bias_alpha
        x = x * tf.where(x > 0, self.pos_slope, self.neg_slope) - self.bias_beta
        return x


class PokeBNN:

    def __init__(self, num_classes=1000) -> None:
        self.activation_bound = [3., 3., 3., 3., 3.]  # need to be determined and frozen during training, by using the method stated in paper
        self.num_classes = num_classes

        self.kwargs = dict(
            input_quantizer=quantizers.PokeSign(precision=1, clip_value=3.),
            kernel_quantizer=quantizers.PokeSign(precision=1, clip_value="dynamic"),
            use_bias=False,
        )

        self.kwargs_init = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_value=self.activation_bound[0]),
            kernel_quantizer=quantizers.PokeSign(precision=8, clip_value="dynamic"),
            use_bias=False,
        )
        self.kwargs_init_depth = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_value=self.activation_bound[1]),
            depthwise_quantizer=quantizers.PokeSign(precision=8, clip_value="dynamic"),
            use_bias=False,
        )
        self.kwargs_SE_1 = dict(
            input_quantizer=quantizers.PokeSign(precision=4, clip_value=self.activation_bound[2]),
            kernel_quantizer=quantizers.PokeSign(precision=4, clip_value="dynamic"),
            use_bias=True
        )
        self.kwargs_SE_2 = dict(
            input_quantizer=quantizers.PokeSign(precision=4, clip_value=self.activation_bound[3], signed=False),
            kernel_quantizer=quantizers.PokeSign(precision=4, clip_value="dynamic", signed=False),
            use_bias=True
        )
        self.kwargs_linear = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_value=self.activation_bound[4]),
            kernel_quantizer=quantizers.PokeSign(precision=8, clip_value="dynamic"),
            use_bias=True
        )

    def SE_4b(self, x: tf.Tensor, c_out: int):
        x = tf.keras.layers.GlobalAveragePooling2D('channels_last', keepdims=True)(x)
        x = lq.layers.QuantConv2D(filters=x.shape[-1] // 8, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                  **self.kwargs_SE_1)(x)
        x = tf.keras.layers.ReLU()(x)

        x = lq.layers.QuantConv2D(filters=c_out, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                  **self.kwargs_SE_2)(x)
        x = tf.math.minimum(tf.math.maximum(x + 3, 0), 6.) / 6

        return x

    def Reshape_Add(self, x, r, expand_cho_op):
        if r is None:
            return x
        if r.shape[-1] < x.shape[-1]:
            ch_mult = x.shape[-1] // r.shape[-1]
            assert expand_cho_op in ['tile', 'zeropad']
            if expand_cho_op == 'tile':
                r = tf.tile(r, multiples=(1, 1, 1, ch_mult))
            if expand_cho_op == 'zeropad':
                pad_size = x.shape[-1] - r.shape[-1]
                r = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, pad_size)), data_format="channels_first")(r)
        elif r.shape[-1] > x.shape[-1]:
            num_ch_avg = r.shape[-1] // x.shape[-1]
            r = r[:, :, :, 0: x.shape[-1] * num_ch_avg]
            width = r.shape[1]
            r = tf.reshape(r, (-1, r.shape[1]*r.shape[2], x.shape[-1]*num_ch_avg))
            r = tf.keras.layers.AveragePooling1D(pool_size=num_ch_avg, strides=num_ch_avg, padding='valid', data_format='channels_first')(r)
            r = tf.reshape(r, (-1, width, width, x.shape[-1]))

        if r.shape[1] != x.shape[1]:
            pool_layer = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
            r = pool_layer(r)

        return x + r

    def conv1x1_1b(self, x, ch, stride, name):
        x = lq.layers.QuantConv2D(filters=ch, kernel_size=(1, 1), name=name, strides=stride, padding="same", pad_values=1,
                                  **self.kwargs)(x)
        return x

    def conv3x3_1b(self, x, ch, stride, name):
        x = lq.layers.QuantConv2D(filters=ch, kernel_size=(3, 3), name=name, strides=stride, padding="same", pad_values=1,
                                  **self.kwargs)(x)
        return x

    def PokeConv(self, x, r1, kernel_size, out_channel, stride=(1, 1), name=None):
        r = x
        if kernel_size == 1:
            x = self.conv1x1_1b(x, out_channel, stride, name + "_conv")
        elif kernel_size == 3:
            x = self.conv3x3_1b(x, out_channel, stride, name + "_conv")
        else:
            raise RuntimeError("No such kernel size available!")

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                               gamma_initializer='zeros' if r is not None else 'ones')(x)
        x = self.Reshape_Add(x, r, "zeropad")
        x = self.Reshape_Add(x, r1, "tile")
        x = DPReLU()(x)
        x = x * self.SE_4b(r, out_channel)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        return x

    def Poke_init(self, x):
        x = lq.layers.QuantConv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding="valid", **self.kwargs_init)(
            x)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = DPReLU()(x)
        x = lq.layers.QuantDepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                           **self.kwargs_init_depth)(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = DPReLU()(x)

        return x

    def build(self, input_shape: tuple) -> tf.keras.Model:
        M = 1
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.Poke_init(inputs)
        for i, strides, features in [
            (0, (1, 1), 64),
            (1, (1, 1), 64),
            (2, (1, 1), 64),
            (3, (2, 2), 128),
            (4, (1, 1), 128),
            (5, (1, 1), 128),
            (6, (1, 1), 128),
            (7, (2, 2), 256),
            (8, (1, 1), 256),
            (9, (1, 1), 256),
            (10, (1, 1), 256),
            (11, (1, 1), 256),
            (12, (1, 1), 256),
            (13, (2, 2), 512),
            (14, (1, 1), 512),
            (15, (1, 1), 512),
        ]:
            r = outputs
            outputs = self.PokeConv(outputs, None, 1, out_channel=features*M, stride=(1, 1),
                                    name="PokeConv_{}_1".format(i))
            outputs = self.PokeConv(outputs, None, 3, out_channel=features*M, stride=strides,
                                    name="PokeConv_{}_2".format(i))
            outputs = self.PokeConv(outputs, r, 1, out_channel=4 * features*M, stride=(1, 1),
                                    name="PokeConv_{}_3".format(i))

        outputs = tf.keras.layers.GlobalAveragePooling2D('channels_last', keepdims=True)(outputs)

        outputs = lq.layers.QuantConv2D(filters=self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid", **self.kwargs_linear)(outputs)
        outputs = tf.keras.layers.Reshape((-1,))(outputs)

        # construct tf model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# model = PokeBNN(num_classes=1000).build(input_shape=(224, 224, 3))
# lq.models.summary(model, print_fn=None, include_macs=True)
# y = model.predict(x=np.random.random((2, 224, 224, 3)))  # run inference
# print(y.shape)
#
# with open("./Poke_model/PokeBNN_1x_no_DP.tflite", "wb") as file:
#     byte = lce.convert_keras_model(model)
#     file.write(byte)
