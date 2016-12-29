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
from new_audio_utils import gen_audio_phonemes_pairs


def train(lstm_size=1000, z_size=100, batch_size=32, fc_size=400,
          checkpoint_dir="vrnn_checkpoints", learning_rate=0.001, clip_grad=5.0,
          num_epochs=50, save_every=5):
    num_steps = 40
    in_dim = 200

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    input_ = Input(batch_shape=(batch_size, num_steps, in_dim))

    # Input but shifed by one-time step
    input_shift = Input(batch_shape=(batch_size, num_steps, in_dim))
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
    hidden_to_out = TimeDistributed(Dense(in_dim))(hidden)
    samples_to_out = TimeDistributed(Dense(in_dim))(samples)
    hid_merge_samples = merge([hidden_to_out, samples_to_out], mode="sum")
    out_mu = TimeDistributed(Dense(in_dim, activation="tanh"))(hid_merge_samples)
    out_log_sigma = TimeDistributed(Dense(in_dim, activation="relu"))(hid_merge_samples)

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
                                        n_epochs=save_every)
    callbacks_list = [checkpoint]

    adam = Adam(lr=learning_rate, clipnorm=clip_grad)
    vae = Model(input=[input_, input_shift], output=out_mu)
    encoder = Model(input=[input_, input_shift], output=Z_mean)

    vae.compile(optimizer=adam, loss=variational_loss)
    vae.fit_generator(
        gen_audio_phonemes_pairs(return_y=True),
        samples_per_epoch=batch_size*545, verbose=2,
        nb_epoch=num_epochs,
        validation_data=gen_audio_phonemes_pairs(return_y=True, path="valid"),
        nb_val_samples=batch_size*147, callbacks=callbacks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument(
        '--checkpoint_dir', nargs="?", default="vrnn_checkpoints", type=str,
        help="Directory to store checkpoints.")
    parser.add_argument(
        '--lstm_size', nargs="?", default=1000, type=int,
        help="Number of hidden lstm units.")
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
    train(z_size=args.z_size, lstm_size=args.lstm_size,
          checkpoint_dir=args.checkpoint_dir, batch_size=args.batch_size,
          fc_size=args.fc_size, clip_grad=args.clip_grad,
          learning_rate=args.learning_rate, num_epochs=args.num_epochs,
          save_every=args.save_every)
