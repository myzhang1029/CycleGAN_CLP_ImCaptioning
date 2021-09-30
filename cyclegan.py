#!/usr/bin/env python3
# coding: utf-8

# Copyright 2019 The TensorFlow Authors.
# Copyright 2021 Zhang Maiyun.
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic TensorFlow v2 CycleGAN Implementation.

Mostly based on https://www.tensorflow.org/tutorials/generative/cyclegan
and https://www.tensorflow.org/tutorials/generative/pix2pix,
with my interpretations and understanding.

This notebook demonstrates unpaired image to image translation using
conditional GAN's, as described in [Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), also
known as CycleGAN. The paper proposes a method that can capture the
characteristics of one image domain and figure out how these characteristics
could be translated into another image domain, all in the absence of any
paired training examples.

CycleGAN uses a cycle consistency loss to enable training without the need for
paired data. In other words, it can translate from one domain to another
without a one-to-one mapping between the source and target domain.

This opens up the possibility to do a lot of interesting tasks like
photo-enhancement, image colorization, style transfer, etc. All you need is the
source and the target dataset (which is simply a directory of images).

# Next steps

This tutorial has shown how to implement CycleGAN starting from the generator
and discriminator implemented in the
[Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix) tutorial.
As a next step, you could try using a different dataset from
[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan).

You could also train for a larger number of epochs to improve the results, or
you could implement the modified ResNet generator used in the
[paper](https://arxiv.org/abs/1703.10593) instead of the U-Net generator used
here.
"""

# ## Set up the input pipeline

# Install the [tensorflow_examples](https://github.com/tensorflow/examples)
# package that enables importing of the generator and the discriminator.


import datetime
import os
from contextlib import ExitStack
from math import ceil
from typing import Any, Dict, Generator, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tqdm import tqdm

# Influential constants
# Penalty coefficient
LAMBDA = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


# ## Pix2pix implementations

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create the variables of the layer."""
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name="offset",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True)

    def call(self, x: Tensor) -> Tensor:
        """Instance normalization."""
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self) -> Dict[str, float]:
        """Get configuration for save and reload."""
        return {"epsilon": self.epsilon}


def downsample(filters: int, size: int, apply_norm: bool = True) -> Model:
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu
    The type of normalization is always instancenorm.

    Parameters
    ----------
      filters : int
        Number of filters
      size : int
        Filter size
      apply_norm : bool
        If True, adds the instancenorm layer

    Returns
    -------
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding="same",
                               kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters: int, size: int, apply_dropout: bool = False) -> Model:
    """Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => Relu
    The type of normalization is always instancenorm.

    Parameters
    ----------
      filters : int
        Number of filters,
      size : int
        Filter size.
      apply_dropout : bool, optional
        If True, adds the dropout layer.

    Returns
    -------
    Model
        Upsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding="same",
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def unet_generator(output_channels: int) -> Model:
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    The type of normalization is always instancenorm.

    Parameters
    ----------
      output_channels : int
        Output channels

    Returns:
      Generator model
    """
    down_stack = [
        downsample(64, 4, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding="same", kernel_initializer=initializer,
        activation="tanh")  # (bs, 256, 256, 3)
    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    save_inputs = inputs
    # Downsampling through the model
    skips = []
    for downsampler in down_stack:
        inputs = downsampler(inputs)
        skips.append(inputs)
    # Reversed from the second last
    skips = skips[-2::-1]
    # Upsampling and establishing the skip connections
    for upsampler, skip in zip(up_stack, skips):
        inputs = upsampler(inputs)
        inputs = concat([inputs, skip])
    inputs = last(inputs)
    return Model(inputs=save_inputs, outputs=inputs)


def discriminator() -> Model:
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    The type of normalization is always instancenorm.
    The target image cannot be an input.

    Returns:
      Discriminator model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
    save_inputs = inputs
    # (bs, 128, 128, 64)
    down1 = downsample(64, 4, False)(inputs)
    # (bs, 64, 64, 128)
    down2 = downsample(128, 4)(down1)
    # (bs, 32, 32, 256)
    down3 = downsample(256, 4)(down2)
    # (bs, 34, 34, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    # (bs, 31, 31, 512)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)
    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    # (bs, 33, 33, 512)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    # (bs, 30, 30, 1)
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)
    return Model(inputs=save_inputs, outputs=last)


# ## Input Pipeline
#
# This tutorial trains a model to translate from images of horses, to images
# of zebras. You can find this dataset and similar ones
# [here](https://www.tensorflow.org/datasets/catalog/cycle_gan).
#
# As mentioned in the [paper](https://arxiv.org/abs/1703.10593), apply random
# jittering and mirroring to the training dataset. These are some of the image
# augmentation techniques that avoids overfitting.
#
# This is similar to what was done in
# [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix)
#
# * In random jittering, the image is resized to `286 x 286` and then randomly
#   cropped to `256 x 256`.
# * In random mirroring, the image is randomly flipped horizontally
#   i.e left to right.

# shape[0] is batch size, but batch size may not divide the total size
image_tensor = tf.TensorSpec(
    shape=[None, IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS],
    dtype=tf.float32
)

any_image_tensor = tf.TensorSpec(
    shape=[None, None, None],
    dtype=tf.uint8
)


def imgen(
    directory: bytes,
    suffix: bytes = b".jpg"
) -> Generator[Tensor, None, None]:
    """Generate images from a directory of images."""
    assert isinstance(directory, bytes)
    for file in os.scandir(directory):
        if file.name.endswith(suffix):
            with open(file.path, "rb") as imgfile:
                img = tf.image.decode_image(imgfile.read())
            # Some images in COCO are greyscale
            if img.shape[2] == 1:
                img = tf.image.grayscale_to_rgb(img)
            yield img


def get_img_dataset(
    directory: str,
    suffix: str = ".jpg"
) -> Tuple[tf.data.Dataset, int]:
    """Create TensorFlow dataset with a directory of images."""
    number = len(list(filter(
        lambda file: file.name.endswith(suffix),
        os.scandir(directory)
    )))
    dataset = tf.data.Dataset.from_generator(
        imgen,
        args=(directory, suffix),
        output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
    )
    return dataset, number


@tf.function(input_signature=(any_image_tensor,))
def preprocess_image_train(image: Tensor) -> Tensor:
    """Preprocess step for training images.

    1. Resize to 286 x 286 x 3.
    2. Randomly crop to 256 x 256 x 3.
    3. Randomly flip.
    4. Normalize the images to [-1, 1].
    """
    image = tf.image.resize(image, [IMG_HEIGHT * 1.12, IMG_WIDTH * 1.12],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    # normalize
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

# ## Loss functions

# In CycleGAN, there is no paired data to train on, hence there is no guarantee
# that the input `x` and the target `y` pair are meaningful during training.
# Thus in order to enforce that the network learns the correct mapping, the
# authors propose the cycle consistency loss.
#
# The discriminator loss and the generator loss are similar to the ones used in
# [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix).


@tf.function
def discriminator_loss(
    loss_obj: tf.losses.Loss, real: Tensor, generated: Tensor
) -> Tensor:
    """Discriminator loss function."""
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


@tf.function
def generator_loss(loss_obj: tf.losses.Loss, generated: Tensor) -> Tensor:
    """Generator loss function."""
    return loss_obj(tf.ones_like(generated), generated)


@tf.function(input_signature=(image_tensor, image_tensor))
def calc_cycle_loss(real_image: Tensor, cycled_image: Tensor) -> Tensor:
    r"""Cycle consistency loss.

    Cycle consistency means the result should be close to the original input.
    For example, if one translates a sentence from English to French, and then
    translates it back from French to English, then the resulting sentence
    should be the same as the original sentence.

    In cycle consistency loss,

    * Image $X$ is passed via generator $G$ that yields generated image
      $\hat{Y}$.
    * Generated image $\hat{Y}$ is passed via generator $F$ that yields cycled
      image $\hat{X}$.
    * Mean absolute error is calculated between $X$ and $\hat{X}$.

    $$
    forward\ cycle\ consistency\ loss: X -> G(X) -> F(G(X)) \sim \hat{X}
    backward\ cycle\ consistency\ loss: Y -> F(Y) -> G(F(Y)) \sim \hat{Y}
    $$
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1


@tf.function(input_signature=(image_tensor, image_tensor))
def identity_loss(real_image: Tensor, same_image: Tensor) -> Tensor:
    r"""Identity loss function.

    As shown above, generator $G$ is responsible for translating image $X$ to
    image $Y$. Identity loss says that, if you fed image $Y$ to generator $G$,
    it should yield the real image $Y$ or something close to image $Y$.

    If you run the zebra-to-horse model on a horse or the horse-to-zebra model
    on a zebra, it should not modify the image much since the image already
    contains the target class.

    $$Identity\ loss = |G(Y) - Y| + |F(X) - X|$$
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


class TrainingMetrics:
    """Encapsulated metrics for this project."""

    def __init__(self) -> None:
        self.metric_gen_g_loss = tf.keras.metrics.Mean(
            "gen_g_loss", dtype=tf.float32)
        self.metric_gen_f_loss = tf.keras.metrics.Mean(
            "gen_f_loss", dtype=tf.float32)
        self.metric_total_cycle_loss = tf.keras.metrics.Mean(
            "total_cycle_loss", dtype=tf.float32)
        self.metric_total_gen_g_loss = tf.keras.metrics.Mean(
            "total_gen_g_loss", dtype=tf.float32)
        self.metric_total_gen_f_loss = tf.keras.metrics.Mean(
            "total_gen_f_loss", dtype=tf.float32)
        self.metric_disc_x_loss = tf.keras.metrics.Mean(
            "total_disc_x_loss", dtype=tf.float32)
        self.metric_disc_y_loss = tf.keras.metrics.Mean(
            "total_disc_y_loss", dtype=tf.float32)

    def __call__(self, metrics: Dict[str, float]) -> None:
        """Update metrics."""
        self.metric_gen_g_loss(metrics["gen_g_loss"])
        self.metric_gen_f_loss(metrics["gen_f_loss"])
        self.metric_total_cycle_loss(metrics["total_cycle_loss"])
        self.metric_total_gen_g_loss(metrics["total_gen_g_loss"])
        self.metric_total_gen_f_loss(metrics["total_gen_f_loss"])
        self.metric_disc_x_loss(metrics["disc_x_loss"])
        self.metric_disc_y_loss(metrics["disc_y_loss"])

    def put_summary(self, step: int) -> None:
        """Put metrics to tf.summary."""
        tf.summary.scalar("Forward Generator Loss",
                          self.metric_gen_g_loss.result(), step=step)
        tf.summary.scalar("Backward Generator Loss",
                          self.metric_gen_f_loss.result(), step=step)
        tf.summary.scalar("Total Cycle Loss",
                          self.metric_total_cycle_loss.result(), step=step)
        tf.summary.scalar("Total Forward Generator Loss",
                          self.metric_total_gen_g_loss.result(), step=step)
        tf.summary.scalar("Total Backward Generator Loss",
                          self.metric_total_gen_f_loss.result(), step=step)
        tf.summary.scalar("X Discriminator Loss",
                          self.metric_disc_x_loss.result(), step=step)
        tf.summary.scalar("Y Discriminator Loss",
                          self.metric_disc_y_loss.result(), step=step)


class CycleGANModel:
    """Encapsulated model."""

    def __init__(self) -> None:
        # ## Reuse the Pix2Pix models
        # Some of the differences are:
        #
        # * CycleGAN uses
        #   [instance normalization](https://arxiv.org/abs/1607.08022) instead
        #   of [batch normalization](https://arxiv.org/abs/1502.03167).
        # * The [CycleGAN paper](https://arxiv.org/abs/1703.10593) uses a
        #   modified `resnet` based generator. This tutorial is using a
        #   modified `unet` generator for simplicity.
        #
        # There are 2 generators (G and F) and 2 discriminators (X and Y) being
        # trained here.
        #
        # * Generator `G` learns to transform image `X` to image `Y`.
        #   $(G: X -> Y)$
        # * Generator `F` learns to transform image `Y` to image `X`.
        #   $(F: Y -> X)$
        # * Discriminator `D_X` learns to differentiate between image `X` and
        #   generated image `X` (`F(Y)`).
        # * Discriminator `D_Y` learns to differentiate between image `Y` and
        #   generated image `Y` (`G(X)`).
        self.generator_g = unet_generator(OUTPUT_CHANNELS)
        self.generator_f = unet_generator(OUTPUT_CHANNELS)
        self.discriminator_x = discriminator()
        self.discriminator_y = discriminator()
        self._init_optimizers()

    def _init_optimizers(self) -> None:
        """Initialize optimizers for all generators and discriminators."""
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Configure the model for training."""
        self.generator_g.compile(*args, **kwargs)
        self.generator_f.compile(*args, **kwargs)
        self.discriminator_x.compile(*args, **kwargs)
        self.discriminator_y.compile(*args, **kwargs)

    @tf.function
    def train_step(self, real_x: Tensor, real_y: Tensor,
                   loss_obj: tf.losses.Loss) -> Dict[str, float]:
        """Train step for CycleGAN.

        Even though the training loop looks complicated, it consists of four
        basic steps:

        * Get the predictions.
        * Calculate the loss.
        * Calculate the gradients using backpropagation.
        * Apply the gradients to the optimizer.
        """
        # pylint: disable=too-many-locals
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)
            # calculate the loss
            gen_g_loss = generator_loss(loss_obj, disc_fake_y)
            gen_f_loss = generator_loss(loss_obj, disc_fake_x)
            total_cycle_loss = (
                calc_cycle_loss(real_x, cycled_x)
                + calc_cycle_loss(real_y, cycled_y)
            )
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = (
                gen_g_loss
                + total_cycle_loss
                + identity_loss(real_y, same_y)
            )
            total_gen_f_loss = (
                gen_f_loss +
                total_cycle_loss +
                identity_loss(real_x, same_x)
            )
            disc_x_loss = discriminator_loss(
                loss_obj, disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(
                loss_obj, disc_real_y, disc_fake_y)
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(
            total_gen_g_loss,
            self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(
            total_gen_f_loss,
            self.generator_f.trainable_variables)
        discriminator_x_gradients = tape.gradient(
            disc_x_loss,
            self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(
            disc_y_loss,
            self.discriminator_y.trainable_variables)
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients, self.generator_g.trainable_variables)
        )
        self.generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients, self.generator_f.trainable_variables)
        )
        self.discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients,
                self.discriminator_x.trainable_variables)
        )
        self.discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients,
                self.discriminator_y.trainable_variables)
        )
        return {
            "gen_g_loss": gen_g_loss,
            "gen_f_loss": gen_f_loss,
            "total_cycle_loss": total_cycle_loss,
            "total_gen_g_loss": total_gen_g_loss,
            "total_gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss
        }

    def fit(self, horses: tf.data.Dataset,
            zebras: tf.data.Dataset,
            *,
            npairs: Optional[int] = None,
            epochs: int = 1,
            ckpt_manager: Optional[tf.train.CheckpointManager] = None,
            summary_path: Optional[str] = None
            ) -> None:
        """Training process."""
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        print(f"Starting training for {epochs} epochs")
        # Metrics for training summary
        metrics = TrainingMetrics()
        with ExitStack() as stack:
            if summary_path is not None:
                # Pick this one as the sample throughout
                sample_horse = next(iter(horses))
                train_summary_writer = tf.summary.create_file_writer(
                    summary_path)
                stack.enter_context(train_summary_writer.as_default())
            for epoch in range(epochs):
                for img_x, img_y in tqdm(
                    tf.data.Dataset.zip((horses, zebras)),
                    total=npairs,
                    unit="batch",
                    desc=f"Epoch {epoch + 1}/{epochs}"
                ):
                    # Update metrics
                    metrics(self.train_step(img_x, img_y, loss_obj))
                if ckpt_manager is not None:
                    ckpt_manager.save()
                # Summaries
                if summary_path is not None:
                    metrics.put_summary(step=epoch)
                    tf.summary.image(
                        "Training original and output",
                        tf.concat((
                            sample_horse,
                            self.generator_g(sample_horse)
                        ), 0),
                        step=epoch)

    def predict(self, *args: Any, **kwargs: Any) -> Tensor:
        """Generate B set predictions for the A set samples."""
        return self.generator_g.predict(*args, **kwargs)

    def rev_predict(self, *args: Any, **kwargs: Any) -> Tensor:
        """Generate A set predictions for the B set samples."""
        return self.generator_f.predict(*args, **kwargs)

    def save(self, dirpath: str) -> None:
        """Save the model. `dirpath` must be a directory."""
        self.generator_g.save_weights(os.path.join(dirpath, "geng.tf"))
        self.generator_f.save_weights(os.path.join(dirpath, "genf.tf"))
        self.discriminator_x.save_weights(os.path.join(dirpath, "disx.tf"))
        self.discriminator_y.save_weights(os.path.join(dirpath, "disy.tf"))

    def load(self, dirpath: str) -> None:
        """Reload a model. `dirpath` must be a directory."""
        self.generator_g.load_weights(os.path.join(dirpath, "geng.tf"))
        self.generator_f.load_weights(os.path.join(dirpath, "genf.tf"))
        self.discriminator_x.load_weights(os.path.join(dirpath, "disx.tf"))
        self.discriminator_y.load_weights(os.path.join(dirpath, "disy.tf"))


def get_checkpoint(cgmodel: CycleGANModel, path: str
                   ) -> tf.train.CheckpointManager:
    """Checkpoints."""
    ckpt = tf.train.Checkpoint(
        generator_g=cgmodel.generator_g,
        generator_f=cgmodel.generator_f,
        discriminator_x=cgmodel.discriminator_x,
        discriminator_y=cgmodel.discriminator_y,
        generator_g_optimizer=cgmodel.generator_g_optimizer,
        generator_f_optimizer=cgmodel.generator_f_optimizer,
        discriminator_x_optimizer=cgmodel.discriminator_x_optimizer,
        discriminator_y_optimizer=cgmodel.discriminator_y_optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    return ckpt_manager


# CLI Variables
# Path to the A set
HORSE_SET_PATH = "./datasets/photo2clp/trainA"
# Path to the B set
ZEBRA_SET_PATH = "./datasets/photo2clp/trainB"
# Path to the test A set
HORSE_TEST_PATH = "./datasets/photo2clp/testA"
# Path to the checkpoints
CHECKPOINT_PATH = "./checkpoints/train"
# Log and summary output
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_LOG_PATH = f"./logs/gradient_tape/{CURRENT_TIME}/train"
TEST_LOG_PATH = f"./logs/gradient_tape/{CURRENT_TIME}/test"
# Path to save the result
MODEL_PATH = "./models"
# Number of training epochs
EPOCHS = 40
# Shuffle buffer size
BUFFER_SIZE = 1000
# Size of training batches
BATCH_SIZE = 1
# Whether to cache or not (large memory required)
CACHE = False


if __name__ == "__main__":
    model = CycleGANModel()
    model.compile()
    ckpt_mgr = get_checkpoint(model, CHECKPOINT_PATH)
    # Load dataset
    train_horses, number_train_horses = get_img_dataset(HORSE_SET_PATH)
    train_zebras, number_train_zebras = get_img_dataset(ZEBRA_SET_PATH)
    total_pairs = min(number_train_horses, number_train_zebras)
    # Large memory usage
    if CACHE:
        train_horses = train_horses.cache()
        train_zebras = train_zebras.cache()
    # Preprocess
    train_horses = train_horses.map(
        preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_zebras = train_zebras.map(
        preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    model.fit(
        train_horses,
        train_zebras,
        npairs=ceil(total_pairs/BATCH_SIZE),
        epochs=EPOCHS,
        ckpt_manager=ckpt_mgr,
        summary_path=TRAIN_LOG_PATH
    )
    # Run the trained model on the test dataset
    test_summary_writer = tf.summary.create_file_writer(TEST_LOG_PATH)
    test_horses, _ = get_img_dataset(HORSE_TEST_PATH)
    test_horses = test_horses.map(
        preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).take(5)
    # pylint: disable=not-context-manager
    with test_summary_writer.as_default():
        for inp in test_horses:
            tf.summary.image(
                "Test original and output",
                tf.concat((inp, model.predict(inp)), 0),
                max_outputs=2,
                step=0
            )
    model.save(MODEL_PATH)
