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

Mostly based on https://www.tensorflow.org/tutorials/generative/cyclegan,
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
import time
from typing import Generator, Tuple

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tqdm import tqdm

# Influential constants
# Path to the checkpoints
CHECKPOINT_PATH = "./checkpoints/train"
# Log and summary output
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_LOG_PATH = f"./logs/gradient_tape/{CURRENT_TIME}/train"
TEST_LOG_PATH = f"./logs/gradient_tape/{CURRENT_TIME}/test"
# Shuffle buffer size
BUFFER_SIZE = 1000
# Penalty coefficient
LAMBDA = 10
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
AUTOTUNE = tf.data.AUTOTUNE


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

image_tensor = tf.TensorSpec(
    shape=[1, IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS],
    dtype=tf.float32
)

any_image_tensor = tf.TensorSpec(
    shape=[None, None, None],
    dtype=tf.uint8
)


def imgen(
    directory: bytes,
    suffix: bytes = b".jpg"
) -> Generator[tf.Tensor, None, None]:
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
def preprocess_image_train(image: tf.Tensor) -> tf.Tensor:
    """Preprocess step for training images.

    1. Resize to 286 x 286 x 3.
    2. Randomly crop to 256 x 256 x 3.
    3. Randomly flip.
    4. Normalize the images to [-1, 1].
    """
    image = tf.image.resize(image, [286, 286],
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
    loss_obj: tf.losses.Loss, real: tf.Tensor, generated: tf.Tensor
) -> float:
    """Discriminator loss function."""
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


@tf.function
def generator_loss(loss_obj: tf.losses.Loss, generated: tf.Tensor) -> float:
    """Generator loss function."""
    return loss_obj(tf.ones_like(generated), generated)


@tf.function(input_signature=(image_tensor, image_tensor))
def calc_cycle_loss(real_image: tf.Tensor, cycled_image: tf.Tensor) -> float:
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
def identity_loss(real_image: tf.Tensor, same_image: tf.Tensor) -> float:
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

    def __init__(self):
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

    def __call__(
        self,
        gen_g_loss: float,
        gen_f_loss: float,
        total_cycle_loss: float,
        total_gen_g_loss: float,
        total_gen_f_loss: float,
        disc_x_loss: float,
        disc_y_loss: float
    ):
        """Update metrics."""
        self.metric_gen_g_loss(gen_g_loss)
        self.metric_gen_f_loss(gen_f_loss)
        self.metric_total_cycle_loss(total_cycle_loss)
        self.metric_total_gen_g_loss(total_gen_g_loss)
        self.metric_total_gen_f_loss(total_gen_f_loss)
        self.metric_disc_x_loss(disc_x_loss)
        self.metric_disc_y_loss(disc_y_loss)

    def put_summary(self, step: int):
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

    def __init__(self):
        # ## Import and reuse the Pix2Pix models
        # Import the generator and the discriminator used in
        # [Pix2Pix](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)
        # via the installed
        # [tensorflow_examples](https://github.com/tensorflow/examples)
        # package.
        #
        # The model architecture used in this tutorial is very similar to what was used
        # in [pix2pix](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py).
        # Some of the differences are:
        #
        # * Cyclegan uses
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
        self.generator_g = pix2pix.unet_generator(
            OUTPUT_CHANNELS, norm_type="instancenorm")
        self.generator_f = pix2pix.unet_generator(
            OUTPUT_CHANNELS, norm_type="instancenorm")
        self.discriminator_x = pix2pix.discriminator(
            norm_type="instancenorm", target=False)
        self.discriminator_y = pix2pix.discriminator(
            norm_type="instancenorm", target=False)
        # Initialize the optimizers for all the generators and discriminators.
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, real_x: tf.Tensor, real_y: tf.Tensor,
                   loss_obj: tf.losses.Loss, metrics: TrainingMetrics):
        """Train step for CycleGAN.

        Even though the training loop looks complicated, it consists of five
        basic steps:

        * Get the predictions.
        * Calculate the loss.
        * Calculate the gradients using backpropagation.
        * Apply the gradients to the optimizer.
        * Save metrics
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
        # Update metrics
        metrics(gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss,
                total_gen_f_loss, disc_x_loss, disc_y_loss)

    def train(self, horses: tf.data.Dataset,
              zebras: tf.data.Dataset,
              npairs: int,
              total_epochs: int,
              ckpt_manager: tf.train.CheckpointManager,
              ):
        """Training process."""
        # Metrics for training summary
        metrics = TrainingMetrics()
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_PATH)
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        print(f"Starting training for {total_epochs} epochs")
        # pylint: disable=not-context-manager
        with train_summary_writer.as_default():
            for epoch in range(total_epochs):
                start = time.time()
                for image_x, image_y in tqdm(
                    tf.data.Dataset.zip((horses, zebras)),
                    total=npairs,
                    unit="img",
                    desc=f"Epoch {epoch + 1}/{total_epochs}"
                ):
                    self.train_step(
                        image_x,
                        image_y,
                        loss_obj,
                        metrics
                    )
                save_pth = ckpt_manager.save()
                print(f"Saving checkpoint for epoch {epoch + 1} at {save_pth}")
                print(f"Took {time.time() - start} s for epoch {epoch + 1}\n")
                # Summaries
                metrics.put_summary(step=epoch)
                tf.summary.image(
                    "Training original and output",
                    tf.concat(
                        (sample_horse, self.generator_g(sample_horse)), 0),
                    max_outputs=2,
                    step=0
                )

    def test(self, horses: tf.data.Dataset):
        """Generate using test dataset."""
        test_summary_writer = tf.summary.create_file_writer(TEST_LOG_PATH)
        horses = horses.map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE
        ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # Run the trained model on the test dataset
        # pylint: disable=not-context-manager
        with test_summary_writer.as_default():
            for inp in horses.take(5):
                tf.summary.image(
                    "Test original and output",
                    tf.concat((inp, self.generator_g(inp)), 0),
                    max_outputs=2,
                    step=0
                )

    def save(self, dirpath: str):
        """Save the model. `dirpath` must be a directory."""
        self.generator_g.save_weights(os.path.join(dirpath, "geng.tf"))
        self.generator_f.save_weights(os.path.join(dirpath, "genf.tf"))
        self.discriminator_x.save_weights(os.path.join(dirpath, "disx.tf"))
        self.discriminator_y.save_weights(os.path.join(dirpath, "disy.tf"))

    def load(self, dirpath: str):
        """Reload a model. `dirpath` must be a directory."""
        self.generator_g.load_weights(os.path.join(dirpath, "geng.tf"))
        self.generator_f.load_weights(os.path.join(dirpath, "genf.tf"))
        self.discriminator_x.load_weights(os.path.join(dirpath, "disx.tf"))
        self.discriminator_y.load_weights(os.path.join(dirpath, "disy.tf"))


def get_checkpoint(cgmodel: CycleGANModel) -> tf.train.CheckpointManager:
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
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, CHECKPOINT_PATH, max_to_keep=5)
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
# Path to save the result
MODEL_PATH = "./models"
# Number of training epochs
EPOCHS = 40
# Whether to cache or not (large memory required)
CACHE = False


if __name__ == "__main__":
    model = CycleGANModel()
    ckpt_mgr = get_checkpoint(model)
    # Load dataset
    train_horses, number_train_horses = get_img_dataset(HORSE_SET_PATH)
    train_zebras, number_train_zebras = get_img_dataset(ZEBRA_SET_PATH)
    total_pairs = min(number_train_horses, number_train_zebras)
    test_horses, _ = get_img_dataset(HORSE_TEST_PATH)
    # Make it an array of one image
    sample_horse = tf.expand_dims(
        preprocess_image_train(next(iter(train_horses))),
        axis=0
    )
    # Large memory usage
    if CACHE:
        train_horses = train_horses.cache()
        train_zebras = train_zebras.cache()
    # Preprocess
    train_horses = train_horses.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_zebras = train_zebras.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    model.train(train_horses, train_zebras, total_pairs, EPOCHS, ckpt_mgr)
    model.test(test_horses)
    model.save(MODEL_PATH)
