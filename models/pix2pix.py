"""
Title: Pix2Pix
Author: [AMSUFV](https://github.com/AMSUFV)
Date created: 2020/06/17
Last modified: 2020/06/17
Description: Pix2Pix implementation subclassing keras.Model.
"""

"""
## Setup
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Create the preprocessing functions
"""


def load_train(image_path):
    stack = load(image_path)
    stack = random_jitter(stack)
    stack = normalize(stack)
    stack = tf.unstack(stack, num=stack.shape[0])

    return stack


def load_test(image_path):
    stack = load(image_path)
    stack = resize(stack, HEIGHT, WIDTH)
    stack = normalize(stack)
    stack = tf.unstack(stack, num=stack.shape[0])

    return stack


def load(img_path):
    file = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(file)

    # images contain both input and target image
    width = tf.shape(image)[1]
    middle = width // 2

    real_image = image[:, :middle, :]
    segmented_image = image[:, middle:, :]

    # images are stacked for better handling
    stack = tf.stack([real_image, segmented_image])
    stack = tf.cast(stack, tf.float32)

    return stack


def resize(stack, height, width):
    return tf.image.resize(
        stack, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


def normalize(stack):
    return (stack / 127.5) - 1


def random_jitter(stack):
    # random jitter is applied by upscaling to 110% the size
    stack = resize(stack, int(WIDTH * 1.1), int(HEIGHT * 1.1))
    # cropping randomnly back to the desired size
    stack = tf.image.random_crop(stack, size=[stack.shape[0], HEIGHT, WIDTH, 3])
    # and performing random mirroring
    if tf.random.uniform(()) > 0.5:
        return tf.image.flip_left_right(stack)
    else:
        return stack


"""
## Create the building blocks
"""


def downscale(x, filters, size, apply_norm=True, slope=0.2):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=size,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        use_bias=False
    )(x)
    if apply_norm:
        x = layers.BatchNormalization()(x)
    return tf.nn.leaky_relu(x, alpha=slope)


def upscale(x, filters, size, apply_dropout=False, rate=0.5):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=size,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    if apply_dropout:
        x = layers.Dropout(rate)(x)
    return tf.nn.relu(x)


"""
## Build the generator
"""


def build_generator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    if assisted:
        input_layer = [layers.Input(shape=input_shape), layers.Input(shape=input_shape)]
        x = layers.concatenate(input_layer)
    else:
        input_layer = x = layers.Input(shape=input_shape)

    down_stack = [
        dict(filters=64, kernel_size=4, apply_norm=False),
        dict(filters=128, kernel_size=4),
        dict(filters=256, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
    ]
    up_stack = [
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4),
        dict(filters=256, kernel_size=4),
        dict(filters=128, kernel_size=4),
        dict(filters=64, kernel_size=4),
    ]

    skips = []
    for block in down_stack:
        x = downscale(
            x,
            block.get('filters'),
            block.get('kernel_size'),
            block.get('apply_norm'),
        )
        skips.append(x)

    skips = reversed(skips[:-1])
    for block, skip in zip(up_stack, skips):
        x = upscale(
            x,
            block.get('filters'),
            block.get('kernel_size'),
            block.get('apply_dropout'),
        )
        x = layers.concatenate([x, skip])

    output_image = layers.Conv2DTranspose(
        filters=3,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )(x)

    return keras.Model(
        inputs=input_layer, outputs=output_image, name="pix2pix_generator"
    )


"""
## Build the discriminator
"""


def build_discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    input_image = layers.Input(shape=input_shape, name="input_image")
    target_image = layers.Input(shape=input_shape, name="target_image")
    x = layers.concatenate([input_image, target_image])

    x = downscale(x, 64, 4, apply_norm=False)
    x = downscale(x, 128, 4)
    x = downscale(x, 256, 4)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        filters=512,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D()(x)

    markov_rf = layers.Conv2D(
        filters=1, kernel_size=4, strides=1, kernel_initializer=initializer
    )(x)

    return keras.Model(inputs=[input_image, target_image], outputs=markov_rf)


"""
## Define Pix2Pix as a `Model` with a custom `train_step`
"""


# noinspection PyMethodOverriding,PyAttributeOutsideInit
class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def call(self, inputs, training=False, disc_output=False):
        outputs = self.generator(inputs, training=training)
        if disc_output:
            dgx = self.discriminator([inputs, gx], training=training)
            outputs = (outputs, dgx)
        return outputs

    def test_step(self, data):
        x, y = data
        gx = self.generator(x, training=False)
        dy = self.discriminator([x, y], training=False)
        dgx = self.discriminator([x, gx], training=False)

        g_loss = self.g_loss_fn(y, gx, dgx)
        d_loss = self.d_loss_fn(dy, dgx)

        return {"g_loss": g_loss, "d_loss": d_loss}

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_d(self, x, gx, y):
        with tf.GradientTape() as t:
            dy = self.discriminator([x, y], training=True)
            dgx = self.discriminator([x, gx], training=True)
            d_loss = self.d_loss_fn(dy, dgx)

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )
        return d_loss

    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x, gx], training=True)
            g_loss = self.g_loss_fn(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )
        return gx, g_loss

    def train_step(self, images):
        x, y = images
        gx, g_loss = self.train_g(x, y)
        d_loss = self.train_d(x, gx, y)
        return {"g_loss": g_loss, "d_loss": d_loss}


"""
Define the losses
"""
bce = keras.losses.BinaryCrossentropy(from_logits=True)


def loss_g(y, gx, dgx):
    loss_dgx = bce(tf.ones_like(dgx), dgx)
    loss_l1 = tf.reduce_mean(tf.abs(y - gx))
    total_loss = loss_dgx + LAMBDA * loss_l1
    return total_loss, loss_l1


def loss_d(dy, dgx):
    loss_y = bce(tf.ones_like(dy), dy)
    loss_gx = bce(tf.zeros_like(dgx), dgx)
    return (loss_y + loss_gx) / 2


"""
## Prepare the dataset
"""

# Variables
BUFFER_SIZE = 400
BATCH_SIZE = 1
LAMBDA = 100
WIDTH = HEIGHT = 256
CHANNELS = 3

url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
path = keras.utils.get_file("facades.tar.gz", origin=url, extract=True)
path = os.path.join(os.path.dirname(path), "facades/")

train = tf.data.Dataset.list_files(path + "train/*.jpg")
train = train.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


"""
## Add a callback to log images during training
"""


class ImageSampling(keras.callbacks.Callback):
    def __init__(self, images, log_dir='logs', freq=1):
        super(ImageSampling, self).__init__()
        self.images = images
        self.freq = freq
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            predictions = []
            for image in images:
                if isinstance(image, tuple):
                    image = image[0]
                predictions.append(self.model.generator(image, training=False) * 0.5 + 0.5)
            predictions = tf.squeeze(predictions)
            with self.writer.as_default():
                tf.summary.image(
                    name='image',
                    data=predictions,
                    step=epoch,
                    max_outputs=predictions.shape[0]
                )

"""
## Train the end-to-end model
"""

gen = build_generator()
disc = build_discriminator()


pix2pix = Pix2Pix(gen, disc)
pix2pix.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    d_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    g_loss_fn=loss_g,
    d_loss_fn=loss_d,
)

pix2pix.fit(train, epochs=5)
