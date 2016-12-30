import os
import argparse

from callbacks import SavePeriodicCheckpoint
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import merge
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from math import pi
from utils import audio_amplitudes_gen
from utils import samples_per_epoch


def train(train_dir, valid_dir=None, lstm_size=1000, num_steps=40,
          z_size=100, batch_size=32, fc_size=400, wav_dim=200,
          checkpoint_dir="vrnn_checkpoints", learning_rate=0.001, clip_grad=5.0,
          num_epochs=50, save_every=5):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    input_ = Input(batch_shape=(batch_size, num_steps, wav_dim))

    # Input but shifed by one-time step
    input_shift = Input(batch_shape=(batch_size, num_steps, wav_dim))
    higher = TimeDistributed(Dense(fc_size, activation="tanh"))(input_)

    # Vanilla LSTM
    hidden = LSTM(lstm_size, return_sequences=True)(higher)

    # Prior on the latent variables (z_{t + 1}) is Dependent on the input
    prior_mean = TimeDistributed(Dense(z_size, activation="tanh"))(hidden)
    prior_log_sigma = TimeDistributed(Dense(z_size, activation="relu"))(hidden)

    # Merge hidden-state and input to form the encoder network.
    hidden_to_z = TimeDistributed(Dense(z_size, activation="relu"))(hidden)
    input_to_z = TimeDistributed(Dense(z_size, activation="relu"))(input_shift)
    hidden_with_input = merge([hidden_to_z, input_to_z], mode="sum")
    Z_mean = TimeDistributed(Dense(z_size, activation="tanh"))(hidden_with_input)
    Z_log_sigma = TimeDistributed(Dense(z_size, activation="relu"))(hidden_with_input)

    def sampling(args):
        Z_mean, Z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, num_steps, z_size))
        return Z_mean + K.exp(Z_log_sigma) * epsilon

    samples = Lambda(sampling)([Z_mean, Z_log_sigma])
    hidden_to_out = TimeDistributed(Dense(wav_dim))(hidden)
    samples_to_out = TimeDistributed(Dense(wav_dim))(samples)
    hid_merge_samples = merge([hidden_to_out, samples_to_out], mode="sum")
    out_mu = TimeDistributed(Dense(wav_dim, activation="tanh"))(hid_merge_samples)
    out_log_sigma = TimeDistributed(Dense(wav_dim, activation="relu"))(hid_merge_samples)

    def gaussian_log_likelihood(y_true, y_pred):
        sigma = K.exp(out_log_sigma)
        term1 = 0.5 * K.square((y_true - y_pred) / sigma)
        term2 = out_log_sigma
        term3 = 0.5 * K.log(2 * pi)
        return K.sum(K.sum(term1 + term2 + term3, axis=-1), axis=-1)

    def KL_divergence(mu1, logsigma1, mu2, logsigma2):
        sigma1 = K.exp(logsigma1)
        sigma2 = K.exp(logsigma2)
        first_term = logsigma2 - logsigma1
        second_term = (K.square(sigma1) + K.square(mu1 - mu2)) / (2 * K.square(sigma2))
        KLD = first_term + second_term - 0.5
        return K.sum(K.sum(KLD, axis=-1), axis=-1)

    def variational_loss(y_true, y_pred):
        return (gaussian_log_likelihood(y_true, y_pred) +
                KL_divergence(Z_mean, Z_log_sigma, prior_mean, prior_log_sigma))

    filepath = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
    checkpoint = SavePeriodicCheckpoint(filepath, monitor='val_loss', verbose=1,
                                        n_epochs=5)
    callbacks_list = [checkpoint]

    adam = Adam(lr=learning_rate, clipnorm=clip_grad)
    vae = Model(input=[input_, input_shift], output=out_mu)
    encoder = Model(input=[input_, input_shift], output=Z_mean)

    vae.compile(optimizer=adam, loss=variational_loss)

    train_gen = audio_amplitudes_gen(
        wavdir=train_dir, num_steps=num_steps, batch_size=batch_size,
        wav_dim=wav_dim)
    n_train_per_epoch = samples_per_epoch(
        wavdir=train_dir, batch_size=batch_size, num_steps=num_steps,
        wav_dim=wav_dim)

    if valid_dir is not None:
        valid_gen = audio_amplitudes_gen(
            wavdir=valid_dir, num_steps=num_steps, batch_size=batch_size,
            wav_dim=wav_dim)
        n_val_per_epoch = samples_per_epoch(
            wavdir=valid_dir, batch_size=batch_size,
            num_steps=num_steps, wav_dim=wav_dim)

        vae.fit_generator(train_gen,
                          samples_per_epoch=batch_size*n_train_per_epoch,
                          verbose=2, nb_epoch=num_epochs,
                          validation_data=valid_gen,
                          nb_val_samples=batch_size*n_val_per_epoch,
                          callbacks=callbacks_list)
    else:
        vae.fit_generator(
            audio_amplitudes_gen(wavdir=train_dir),
            samples_per_epoch=batch_size*n_train_per_epoch, verbose=2,
            nb_epoch=num_epochs, callbacks=callbacks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument(
        '--checkpoint_dir', nargs="?", default="vrnn_checkpoints", type=str,
        help="Directory to store checkpoints.")
    parser.add_argument(
        '--train_dir', nargs="?", default="train_dir", type=str,
        help="Directory that contains train wav files to train.")
    parser.add_argument(
        '--valid_dir', nargs="?", default=None,
        help="Directory that contains validation wav files.")
    parser.add_argument(
        '--lstm_size', nargs="?", default=1000, type=int,
        help="Number of hidden lstm units.")
    parser.add_argument(
        '--num_steps', nargs="?", default=40, type=int,
        help="Number of lstm time-steps.")
    parser.add_argument(
        '--wav_dim', nargs="?", default=200, type=int,
        help="Dimension of the input waveform.")
    parser.add_argument('--z_size', nargs="?", default=100, type=int,
        help="Latent size dimensions.")
    parser.add_argument('--batch_size', nargs="?", default=32, type=int,
        help="Batch size.")
    parser.add_argument('--fc_size', nargs="?", default=400, type=int,
        help="Dimension of the input fully-connected layer before providing as "
        "input to LSTM")
    parser.add_argument('--learning_rate', nargs="?", default=0.001, type=float,
        help="Learning rate of the Adam optimizer.")
    parser.add_argument('--clip_grad', nargs="?", default=5.0, type=float,
        help="Clip the value of gradients above clip_grad to clip_grad")
    parser.add_argument('--num_epochs', nargs="?", default=50, type=int,
        help="Number of epochs")
    parser.add_argument('--save_every', nargs="?", default=5, type=int,
        help="Save the model every save_every number of epochs.")

    args = parser.parse_args()
    train(train_dir=args.train_dir, valid_dir=args.valid_dir,
          z_size=args.z_size, lstm_size=args.lstm_size, num_steps=args.num_steps,
          checkpoint_dir=args.checkpoint_dir, batch_size=args.batch_size,
          fc_size=args.fc_size, clip_grad=args.clip_grad,
          learning_rate=args.learning_rate, num_epochs=args.num_epochs,
          save_every=args.save_every, wav_dim=args.wav_dim)
