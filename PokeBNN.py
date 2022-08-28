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
        self.bias_alpha = self.add_weight("alpha", shape=[1, 1, 1, input_shape[-1]], initializer='zeros', trainable=True)
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
    """ larq implementation of PokeBNN, phase=1 for non-quantized phase for training, phase=2 for normal quantized version."""

    def __init__(self, num_classes=1000, phase=1, act_B=None) -> None:
        self.num_classes = num_classes

        self.kwargs = dict(
            input_quantizer=quantizers.PokeSign(precision=1, clip_way="binary_act", phase=phase),
            kernel_quantizer=quantizers.PokeSign(precision=1, clip_way="weight", phase=phase),
            use_bias=False,
        )

        self.kwargs_init = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_way="mul_act", phase=phase, clip_B = act_B["init_conv"]),
            kernel_quantizer=quantizers.PokeSign(precision=8, clip_way="weight", phase=phase),
            use_bias=False,
        )

        self.kwargs_init_depth = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_way="mul_act", phase=phase, clip_B = act_B["init_depth_conv"]),
            depthwise_quantizer=quantizers.PokeSign(precision=8, clip_way="weight", phase=phase),
            use_bias=False,
        )

        self.kwargs_SE_1 = []
        for i in range(48):
            self.kwargs_SE_1.append(dict(
                input_quantizer=quantizers.PokeSign(precision=4, clip_way="mul_act", phase=phase, clip_B = act_B["SE_1"][i]),
                kernel_quantizer=quantizers.PokeSign(precision=4, clip_way="weight", phase=phase),
                use_bias=True
            ))

        self.kwargs_SE_2 = []
        for i in range(48):
            self.kwargs_SE_2.append(dict(
                input_quantizer=quantizers.PokeSign(precision=4, clip_way="mul_act", signed=False, phase=phase, clip_B = act_B["SE_2"][i]),
                kernel_quantizer=quantizers.PokeSign(precision=4, clip_way="weight", signed=False, phase=phase),
                use_bias=True
            ))

        self.kwargs_linear = dict(
            input_quantizer=quantizers.PokeSign(precision=8, clip_way="mul_act", phase=phase, clip_B = act_B["linear"]),
            kernel_quantizer=quantizers.PokeSign(precision=8, clip_way="weight", phase=phase),
            use_bias=True
        )

    def SE_4b(self, x: tf.Tensor, c_out: int, idx_num: int):

        x = tf.keras.layers.GlobalAveragePooling2D('channels_last', keepdims=True)(x)
        x = lq.layers.QuantConv2D(filters=x.shape[-1] // 8, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="SE_1_{}".format(idx_num),
                                  **(self.kwargs_SE_1[idx_num]))(x)
        x = tf.keras.layers.ReLU()(x)

        x = lq.layers.QuantConv2D(filters=c_out, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="SE_2_{}".format(idx_num),
                                  **(self.kwargs_SE_2[idx_num]))(x)
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

    def PokeConv(self, x, r1, kernel_size, out_channel, stride=(1, 1), name=None, idx=0):
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
        x = x * self.SE_4b(r, out_channel, idx)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        return x

    def Poke_init(self, x):
        x = lq.layers.QuantConv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding="valid", name="init_conv", **self.kwargs_init)(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = DPReLU()(x)
        x = lq.layers.QuantDepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 3), strides=(1, 1), padding="same", name="init_depth_conv",
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
                                    name="PokeConv_{}_1".format(i), idx=i*3)
            outputs = self.PokeConv(outputs, None, 3, out_channel=features*M, stride=strides,
                                    name="PokeConv_{}_2".format(i), idx=i*3+1)
            outputs = self.PokeConv(outputs, r, 1, out_channel=4 * features*M, stride=(1, 1),
                                    name="PokeConv_{}_3".format(i), idx=i*3+2)

        outputs = tf.keras.layers.GlobalAveragePooling2D('channels_last')(outputs)

        # outputs = lq.layers.QuantConv2D(filters=self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="linear", **self.kwargs_linear)(outputs)
        # outputs = tf.keras.layers.Reshape((-1,))(outputs)

        outputs = lq.layers.QuantDense(units=self.num_classes, name="linear", activation="softmax", **self.kwargs_linear)(outputs)

        # construct tf model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


if __name__ == "__main__":
    # pick 3.0 as default non-binary act bound B
    act_B = {"init_conv": 3.0, "init_depth_conv": 3.0, "linear": 3.0, "SE_1": [3.0]*48, "SE_2": [3.0]*48}
    model = PokeBNN(num_classes=1000, phase=2, act_B=act_B).build(input_shape=(224, 224, 3))
    lq.models.summary(model, print_fn=None, include_macs=True)
    y = model(np.random.random((2, 224, 224, 3)), training=False)  # run inference
    print(y.shape)

    # act_B["init_conv"] = model.get_layer("init_conv").input_quantizer.clip_B
    # act_B["init_depth_conv"] = model.get_layer("init_depth_conv").input_quantizer.clip_B
    # act_B["linear"] = model.get_layer("linear").input_quantizer.clip_B
    # for i in range(48):
    #     act_B["SE_1"][i] = model.get_layer("SE_1_{}".format(i)).input_quantizer.clip_B
    #     act_B["SE_2"][i] = model.get_layer("SE_2_{}".format(i)).input_quantizer.clip_B

    with open("../Poke_model/model_file/PokeBNN_1x.tflite", "wb") as file:
        byte = lce.convert_keras_model(model)
        file.write(byte)




