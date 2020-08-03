from __future__ import print_function, division
import os
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from segmentation_models.utils import freeze_model
import scipy.stats as st
from commons import get_jnd_map
from skimage.io import imread
from skimage.color import rgb2yuv
from layers import ReflectionPadding2D, conv_bn_relu, res_conv, dconv_bn_nolinear
from tensorflow.python.keras.applications import vgg16
from scipy_optimizer import ScipyOptimizer, GradientObserver
import argparse


class DeepBatterySaver:
    def __init__(self, source_path, learning_rate):
        # Get input image and its JND map
        lamada = 30
        img = imread(source_path)
        img_y = rgb2yuv(img)[:, :, 0]
        jnd = get_jnd_map(img_y) / 255 * lamada
        jnd = np.expand_dims(jnd, axis=0)
        self.jnd = np.expand_dims(jnd, axis=3)

        img = img / 127.5 - 1
        self.img = np.expand_dims(img, axis=0)

        # Set input shape
        _, self.img_rows, self.img_cols, self.channels = self.img.shape
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Get content feature
        self.pretrained_model = self.build_backbone()

        # Build network
        img = Input(shape=self.img_shape)
        self.deep_battery_saver = self.build_model()
        pre_img, diff_img = self.deep_battery_saver(img)

        # Set parameters
        self.tv_loss_w = 0.00004
        self.content_loss_w = 0.025
        self.color_loss_w = 0.005
        self.power_loss_w = 100
        self.pixel_loss_w = 10

        self.combined = Model(inputs=img, outputs=[pre_img, pre_img, pre_img, pre_img, diff_img])
        self.combined.compile(loss=["mse",
                                    self.color_loss,
                                    self.content_loss,
                                    self.total_variation_loss,
                                    "mse"],
                              loss_weights=[ self.pixel_loss_w,
                                             self.color_loss_w,
                                             self.content_loss_w,
                                             self.tv_loss_w,
                                             self.power_loss_w],
                              optimizer=GradientObserver())
        self.opt = ScipyOptimizer(self.combined)
        self.opt.learning_rate = learning_rate

    def build_backbone(self):
        backbone = vgg16.VGG16(weights='imagenet', include_top=False)
        freeze_model(backbone)
        return Model(backbone.input, [backbone.layers[16].output])

    def content_loss(self, y, x):
        y_feature = self.pretrained_model(y)
        x_feature = self.pretrained_model(x)
        return tf.reduce_sum(tf.pow(y_feature - x_feature, 2)) / 2

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def blur(self, x):
        kernel_var = self.gauss_kernel(21, 3, 3)
        return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

    def color_loss(self, y, x):
        yTrue_blur = self.blur(y)
        yPred_blur = self.blur(x)
        loss_color = tf.reduce_sum(tf.pow(yTrue_blur - yPred_blur, 2)) / 2
        return loss_color

    def pixel_loss(self, y, x):
        loss_pixel = tf.reduce_sum(tf.pow(y - x, 2))
        return loss_pixel

    def total_variation_loss(self, y, x):
        img_rows, img_cols = self.img_rows, self.img_cols
        a = K.square(
            x[:, :img_rows - 1, :img_cols - 1, :] -
            x[:, 1:, :img_cols - 1, :]
        )
        b = K.square(
            x[:, :img_rows - 1, :img_cols - 1, :] -
            x[:, :img_rows - 1, 1:, :]
        )
        c = K.square(
            y[:, :img_rows - 1, :img_cols - 1, :] -
            y[:, 1:, :img_cols - 1, :]
        )
        d = K.square(
            y[:, :img_rows - 1, :img_cols - 1, :] -
            y[:, :img_rows - 1, 1:, :]
        )
        return -(K.sum(K.pow(a + b, 1.25)) + K.sum(K.pow(c + d, 1.25))) / 2

    def power_loss(self, y, x, z):
        y_Y = tf.image.rgb_to_yuv(y)[:, :, :, 0]
        x_Y = tf.image.rgb_to_yuv(x)[:, :, :, 0]
        diff = y_Y - x_Y
        loss_luminiance = tf.reduce_sum(tf.pow(diff - z, 2))
        return loss_luminiance

    def build_model(self):
        def get_rgb2y(inputs):
            return tf.image.rgb_to_yuv(inputs)[:, :, :, 0:1]

        def get_y2rgb(inputs):
            x, y = inputs
            uv = tf.image.rgb_to_yuv(x)[:, :, :, 1:3]
            return tf.image.yuv_to_rgb(tf.concat([y, uv], axis=3))

        def get_diff(inputs):
            x, y = inputs
            return tf.subtract(x, y)

        x = Input(shape=self.img_shape)
        x_y = Lambda(get_rgb2y, name='get_rgb2y')(x)
        a = ReflectionPadding2D(padding=(40, 40), input_shape=(256, 256, 1))(x_y)
        a = conv_bn_relu(32, 9, 9, stride=(1, 1))(a)
        a = conv_bn_relu(64, 9, 9, stride=(2, 2))(a)
        a = conv_bn_relu(128, 3, 3, stride=(2, 2))(a)

        for i in range(5):
            a = res_conv(128, 3, 3)(a)
        a = dconv_bn_nolinear(64, 3, 3)(a)
        a = dconv_bn_nolinear(32, 3, 3)(a)
        y = dconv_bn_nolinear(1, 9, 9, stride=(1, 1), activation="tanh")(a)

        diff = Lambda(get_diff, name='get_diff')([x_y, y])
        output = Lambda(get_y2rgb, name='get_y2rgb')([x, y])

        return Model(x, [output, diff])

    def online_training(self, epochs, eta, target_path):
        loss, _ = self.opt.fit(self.img,
                               [self.img, self.img, self.img, self.img, self.jnd],
                               epochs=epochs,
                               verbose=1,
                               method='l-bfgs-b',
                               target_path=target_path,
                               validation_data=(self.img, eta),
                               generator_model=self.deep_battery_saver)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--power_level', type=int, default=0.8)
    parser.add_argument('--learning_rate', type=int, default=0.05)
    parser.add_argument('--source_path', type=str, default='./test/Lena.jpg')
    parser.add_argument('--target_path', type=str, default='./results')
    opt = parser.parse_args()

    model = DeepBatterySaver(source_path=opt.source_path, learning_rate=opt.learning_rate)
    model.online_training(epochs=opt.num_epochs, eta=opt.power_level, target_path=opt.target_path)