# src/models_srgan.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Conv2D,
    PReLU,
    BatchNormalization,
    UpSampling2D,
    LeakyReLU,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.layers import add
from tensorflow.keras.applications.vgg19 import VGG19


def res_block(x):
    """
    Residual block used in the SRGAN generator.
    """
    shortcut = x

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)

    x = add([shortcut, x])
    return x


def upscale_block(x):
    """
    Upsampling block: conv + upsampling + PReLU.
    """
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = UpSampling2D(size=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def create_generator(lr_shape, num_res_blocks=16):
    """
    Build SRGAN generator that upsamples LR image to HR.

    lr_shape: shape of LR input (H, W, C)
    num_res_blocks: number of residual blocks
    """
    lr_input = Input(shape=lr_shape)

    x = Conv2D(64, (9, 9), padding="same")(lr_input)
    x = PReLU(shared_axes=[1, 2])(x)
    residual = x

    for _ in range(num_res_blocks):
        x = res_block(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = add([x, residual])

    x = upscale_block(x)
    x = upscale_block(x)

    sr_output = Conv2D(3, (9, 9), padding="same")(x)

    model = Model(inputs=lr_input, outputs=sr_output, name="srgan_generator")
    return model


def discriminator_block(x, filters, strides=1, use_bn=True):
    """
    Block used in discriminator: Conv -> (BN) -> LeakyReLU.
    """
    x = Conv2D(filters, (3, 3), strides=strides, padding="same")(x)
    if use_bn:
        x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def create_discriminator(hr_shape):
    """
    Build SRGAN discriminator.

    hr_shape: shape of HR input (H, W, C)
    """
    hr_input = Input(shape=hr_shape)
    df = 64

    x = discriminator_block(hr_input, df, use_bn=False)
    x = discriminator_block(x, df, strides=2)
    x = discriminator_block(x, df * 2)
    x = discriminator_block(x, df * 2, strides=2)
    x = discriminator_block(x, df * 4)
    x = discriminator_block(x, df * 4, strides=2)
    x = discriminator_block(x, df * 8)
    x = discriminator_block(x, df * 8, strides=2)

    x = Flatten()(x)
    x = Dense(df * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    validity = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=hr_input, outputs=validity, name="srgan_discriminator")
    return model


def build_vgg_feature_extractor(hr_shape):
    """
    Build a VGG19-based feature extractor for perceptual loss.
    """
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    feature_extractor = Model(
        inputs=vgg.inputs,
        outputs=vgg.layers[10].output,
        name="vgg19_features",
    )
    feature_extractor.trainable = False
    return feature_extractor
