import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Input, LeakyReLU, Conv2D, Dense, Dropout, LSTM, \
                                    BatchNormalization, Flatten, concatenate, Softmax, \
                                    ReLU, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from datasets.tfrecord_utils import _parse_function

class MultiPathBase(ABC):
    """ Base class for MultiPath variants."""

    def __init__(self,
                 num_timesteps=25,
                 num_hist_timesteps=5,
                 lr_min=1e-4,
                 lr_max=1e-3,
                 lr_max_decay=1e-2,
                 lr_period=10,
                 use_context=True):

        self.num_timesteps = num_timesteps
        self.history_timesteps = num_hist_timesteps
        self.image_shape = (500, 500, 3)
        self.past_state_shape = (self.history_timesteps, 4)

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay = lr_max_decay
        self.lr_period = lr_period

        self.use_context = use_context # if False, give a zeroed out image for training/prediction.

        # We use the minimum learning rate for the first epoch for learning rate "warmup".
        optimizer = SGD(lr=self.lr_min,
                        momentum=0.9,
                        nesterov=True,
                        clipnorm=10.)

        self.model, self.loss_function, self.metric_function = self._create_model()
        self.model.compile(loss=self.loss_function,
                           metrics=self.metric_function,
                           optimizer=optimizer)
        self.trained = False
        print(self.model.summary())
        print(f"Training with context: {self.use_context}")

    @abstractmethod
    def _create_model(self):
        """ Exact architecture to be determined by child class.
            Use common backbone and adjust "head" as needed.
            This should return a model, loss function, and metric function
            as observed in the __init__ function."""
        raise NotImplementedError

    @abstractmethod
    def _extract_gmm_params(self, gmm_pred):
        """ Given a prediction batch from the child network, returns
            a list of dictionaries with length = len(gmm_pred) that
            contains the parameters of the GMM.

            Each dictionary should have keys corresponding to mode_id.
            Each mode_id key should map to a sub-directionary with the keys/values:
                * mode_probability: a scalar probability for that mode
                * mus: GMM mean array across time with shape (self.num_timesteps, 2)
                * sigmas: GMM covariance array across time with shape (self.num_timesteps, 2, 2)
        """
        raise NotImplementedError

    @staticmethod
    def resnet_backbone(image_input, past_state_input, l2_penalty=1e-6):
        """ Provides a common ResNet50 backbone to generate
            image/state features for further processing. """

        # Reference to save LeakyReLU layer:
        # https://www.gitmemory.com/issue/keras-team/keras/6532/481666883
        my_leakyrelu = LeakyReLU(alpha=0.3)
        my_leakyrelu.__name__ = 'lrelu'

        base_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_shape=image_input.shape[1:],
                              pooling=False)
        layer_output = base_model.get_layer(name='conv4_block1_out').output
        base_model = Model(inputs=base_model.input,
                           outputs=layer_output,
                           name='resnet_c4')

        # Image Resolution vs. Stage in Resnet50:
        # Input size: 500 x 500 x 3
        # conv2: 125 x 125 x 256
        # conv3: 63 x 63 x 512
        # conv4: 32 x 32 x 1024 (truncated after first block)
        # conv5: 16 x 16 x 2048 (unused)

        for layer in base_model.layers:
            # Freeze all layers except for conv3 and conv4.
            if 'conv3' in layer.name or 'conv4' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        x = base_model(image_input)

        x = Conv2D(8, (1,1),
                   kernel_regularizer=l2(l2_penalty),
                   bias_regularizer=l2(l2_penalty))(x)
        x = ReLU()(x)
        x = Flatten()(x)

        state_input = LSTM(16,
                           kernel_regularizer=l2(l2_penalty),
                           bias_regularizer=l2(l2_penalty))(past_state_input)
        y = concatenate([x, state_input])
        y = BatchNormalization()(y)

        for _ in range(2):
            y = Dense(512,
                      activation=my_leakyrelu,
                      kernel_regularizer=l2(l2_penalty),
                      bias_regularizer=l2(l2_penalty))(y)
            y = Dropout(0.2)(y)

        return y

    @staticmethod
    def preprocess_entry(entry, use_image=True):
        """ Prepares a batch of features = (images, past_states) and
            labels = xy trajectory given an entry from a TF Dataset.
        """
        if use_image:
            img = preprocess_input(tf.cast(entry['image'], dtype=tf.float32))
        else:
            img = tf.zeros(shape=entry['image'].shape, dtype=tf.float32)

        future_xy = tf.cast(entry['future_poses_local'][:, :, :2], dtype=tf.float32)

        # Concatenate relative timestamp with poses.  Flip the ordering so
        # the first entry is the earliest timestamp and
        # the last entry is the most recent timestamp.
        past_states = tf.concat([tf.expand_dims(entry['past_tms'][:, ::-1], -1),
                                 entry['past_poses_local'][:, ::-1, :]], -1)
        past_states = tf.cast(past_states, dtype=tf.float32)

        return img, past_states, future_xy

    def fit(self,
            train_set,
            val_set,
            logdir=None,
            log_epoch_freq=2,
            save_epoch_freq=10,
            num_epochs=100,
            batch_size=32):

        if logdir is None:
            raise ValueError("Need to provide a logdir for TensorBoard logging and model saving.")
        os.makedirs(logdir, exist_ok=True)

        # Note: the following pipeline does two levels of shuffling:
        # file order shuffling ("global"/"macro" level)
        # dataset instance shuffling ("local" level)
        files = tf.data.Dataset.from_tensor_slices(train_set)
        files = files.shuffle(buffer_size=len(train_set),
                              reshuffle_each_iteration=True)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                                   cycle_length=2,
                                   block_length=16)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(10 * batch_size,
                                  reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

        # Val dataset only used to evaluate loss/metrics so we don't require random or large batches.
        val_dataset = tf.data.TFRecordDataset(val_set)
        val_dataset = val_dataset.map(_parse_function)
        val_dataset = val_dataset.batch(2)  # making this smaller to reduce potential OOM issues

        lr_max = self.lr_max

        tensorboard = TensorBoard(log_dir=logdir,
                                  histogram_freq=0,
                                  write_graph=False)
        tensorboard.set_model(self.model)

        for epoch in range(num_epochs + 1):
            print('Epoch {} started at {}'.format(epoch, datetime.now()))

            losses = []
            ades = []

            for entry in dataset:
                img, past_states, future_xy = self.preprocess_entry(entry, use_image=self.use_context)
                batch_loss, batch_ade = self.model.train_on_batch([img, past_states], future_xy)
                losses.append(batch_loss)
                ades.append(batch_ade)

            epoch_loss = np.mean(losses)
            epoch_ade = np.mean(ades)
            print('\tTrain Loss: {}, Train ADE: {}'.format(epoch_loss, epoch_ade))

            if log_epoch_freq and epoch % log_epoch_freq == 0:
                val_losses = []
                val_ades = []

                for entry in val_dataset:
                    img, past_states, future_xy = self.preprocess_entry(entry, use_image=self.use_context)
                    batch_loss, batch_ade = self.model.test_on_batch(
                        [img, past_states], future_xy)
                    val_losses.append(batch_loss)
                    val_ades.append(batch_ade)

                val_epoch_loss = np.mean(val_losses)
                val_epoch_ade = np.mean(val_ades)
                print('\t Val Loss: {}, Val ADE: {}'.format(val_epoch_loss, val_epoch_ade))

                epoch_summary = {'lr': K.get_value(self.model.optimizer.lr),
                                 'loss': epoch_loss,
                                 'ADE': epoch_ade,
                                 'val_loss': val_epoch_loss,
                                 'val_ADE': val_epoch_ade}
                tensorboard.on_epoch_end(epoch, epoch_summary)

            if save_epoch_freq and epoch % save_epoch_freq == 0:
                filename = logdir + '{0:05d}_epochs'.format(epoch)
                self.save_weights(filename)
                print('Saving model weights at epoch {} to {}.'.format(epoch, filename))

            # Update learning rate for the next epoch.
            if epoch % self.lr_period == 0:
                lr_max = self.lr_max / (1 + self.lr_max_decay * epoch) # lr max decay
            cos_arg = np.pi * (epoch % self.lr_period) / self.lr_period
            lr = self.lr_min + 0.5 * (lr_max - self.lr_min) * (1. + np.cos(cos_arg))
            K.set_value(self.model.optimizer.lr, lr)

            # Standard step-wise decay, non-cyclical.
            # lr = self.lr_max / (1 + self.lr_max_decay * epoch) # lr max decay
            # K.set_value(self.model.optimizer.lr, lr)

        tensorboard.on_train_end(None)
        self.trained = True

    def predict(self, dataset):
        """ Given a dataset, returns the GMM predictions for further analysis.
            This excludes the original image to reduce memory footprint.
        """
        predict_dict = {}

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(32)

        for entry in tqdm(dataset):
            keys = [f"{tf.compat.as_str(x)}_{tf.compat.as_str(y)}"
                    for (x, y) in zip(entry['sample'].numpy(), entry['instance'].numpy())
                   ]
            types = [tf.compat.as_str(x) for x in entry['type'].numpy()]
            vxs = np.ravel(tf.cast(entry['velocity'], dtype=tf.float32).numpy())
            wzs = np.ravel(tf.cast(entry['yaw_rate'], dtype=tf.float32).numpy())
            accs = np.ravel(tf.cast(entry['acceleration'], dtype=tf.float32).numpy())
            poses = tf.cast(entry['pose'], dtype=tf.float32).numpy()
            future_states = tf.concat([tf.expand_dims(entry['future_tms'], -1),
                                       entry['future_poses_local']], -1)
            future_states = tf.cast(future_states, dtype=tf.float32)

            img, past_states, _ = self.preprocess_entry(entry, use_image=self.use_context)
            gmm_pred = self.model.predict_on_batch([img, past_states])  # raw prediction tensor
            gmm_dicts = self._extract_gmm_params(gmm_pred)  # processed prediction as a dict

            zip_obj = zip(keys, types, vxs, wzs, accs, poses, past_states,
                          future_states, gmm_dicts)

            for key, atype, vx, wz, acc, pose, psts, fsts, gmm_dict in zip_obj:
                predict_dict[key] = {
                    'type': atype,
                    'velocity': vx,
                    'yaw_rate': wz,
                    'acceleration': acc,
                    'pose': pose,
                    'past_traj': psts.numpy(),
                    'future_traj': fsts.numpy(),
                    'gmm_pred': gmm_dict
                }

        return predict_dict

    def predict_instance(self, image_raw, past_states):
        if len(image_raw.shape) == 3:
            image_raw = np.expand_dims(image_raw, 0)

        if self.use_context:
            img = preprocess_input(tf.cast(image_raw, dtype=tf.float32))
        else:
            img = tf.zeros(shape=image_raw.shape, dtype=tf.float32)

        if len(past_states.shape) == 2:
            past_states = np.expand_dims(past_states, 0)
        past_states = tf.cast(past_states, dtype=tf.float32)

        gmm_pred = self.model.predict_on_batch([img, past_states])  # raw prediction tensor
        gmm_dicts = self._extract_gmm_params(gmm_pred)  # processed prediction as a dict

        return gmm_dicts[0]

    def save_weights(self, path):
        path = path if '.h5' in path else (path + '.h5')
        self.model.save_weights(path)

    def load_weights(self, path):
        path = path if '.h5' in path else (path + '.h5')
        self.model.load_weights(path)

    def save_model(self, model_dir):
        """ Save the entire the model for deployment.
            Default format is a Tensorflow SavedModel. """
        self.model.save(model_dir)
