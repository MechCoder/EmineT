import os
from keras.layers import Input, Lambda, LSTM, Dense, TimeDistributed, merge, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from callbacks import SavePeriodicCheckpoint

from math import pi
from config import parse_args
from utils import gen_audio_phonemes_pairs
from utils import audio_amplitudes_gen

def train(train_dir, lyr_dir, valid_dir=None, lyr_valid_dir=None,
          lstm_size=1000, num_steps=40, step_shift=0,
          z_dim=100, batch_size=32, fc_dim=400, wav_dim=200,
          checkpoint_dir="vrnn_checkpoints", learning_rate=0.001, clip_grad=5.0,
          num_epochs=50, save_every=5):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    vae = build_vrnn(lstm_size=lstm_size, num_steps=num_steps, z_dim=z_dim,
                     batch_size=batch_size, fc_dim=fc_dim, wav_dim=wav_dim,
                     learning_rate=learning_rate, clip_grad=clip_grad)
    #
    # input_ = Input(batch_shape=(batch_size, num_steps, in_dim))
    # input_shift = Input(batch_shape=(batch_size, num_steps, in_dim))
    # ph_input = Input(batch_shape=(batch_size, num_steps, phoneme_length))
    #
    # # Merge input from phonemes and audio.
    # cbow = TimeDistributed(Dense(phonemes_embed_size))(ph_input)
    # input_layer2 = TimeDistributed(Dense(phonemes_embed_size, activation="relu"))(input_)
    # lyrics_plus_audio = merge([input_layer2, cbow], mode="sum")
    #
    # # Vanilla LSTM
    # hidden = LSTM(hidden_size, return_sequences=True)(lyrics_plus_audio)
    #
    # # Prior on the latent variables (z_{t + 1}) is Dependent on the input
    # prior_mean = TimeDistributed(Dense(z_dim, activation="tanh"))(hidden)
    # prior_log_sigma = TimeDistributed(Dense(z_dim, activation="relu"))(hidden)
    #
    # # Merge hidden-state and input to form the encoder network.
    # hidden_to_z = TimeDistributed(Dense(z_dim, activation="relu"))(hidden)
    # input_to_z = TimeDistributed(Dense(z_dim, activation="relu"))(input_shift)
    # hidden_with_input = merge([hidden_to_z, input_to_z], mode="sum")
    # Z_mean = TimeDistributed(Dense(z_dim, activation="tanh"))(hidden_with_input)
    # Z_log_sigma = TimeDistributed(Dense(z_dim, activation="relu"))(hidden_with_input)
    #
    # def sampling(args):
    #     Z_mean, Z_log_sigma = args
    #     epsilon = K.random_normal(shape=(batch_size, num_steps, z_dim))
    #     return Z_mean + K.exp(Z_log_sigma) * epsilon
    #
    # samples = Lambda(sampling)([Z_mean, Z_log_sigma])
    #
    # hidden_to_out = TimeDistributed(Dense(in_dim))(hidden)
    # samples_to_out = TimeDistributed(Dense(in_dim))(samples)
    # hid_merge_samples = merge([hidden_to_out, samples_to_out], mode="sum")
    # out_mu = TimeDistributed(Dense(in_dim, activation="tanh"))(hid_merge_samples)
    # out_log_sigma = TimeDistributed(Dense(in_dim, activation="relu"))(hid_merge_samples)
    #
    # def gaussian_log_likelihood(y_true, y_pred):
    #     sigma = K.exp(out_log_sigma)
    #     term1 = 0.5 * K.square((y_true - y_pred) / sigma)
    #     term2 = out_log_sigma
    #     term3 = 0.5 * K.log(2 * pi)
    #     return K.sum(K.sum(term1 + term2 + term3, axis=-1), axis=-1)
    #
    # def KL_divergence(mu1, logsigma1, mu2, logsigma2):
    #     sigma1 = K.exp(logsigma1)
    #     sigma2 = K.exp(logsigma2)
    #     first_term = logsigma2 - logsigma1
    #     second_term = (K.square(sigma1) + K.square(mu1 - mu2)) / (2 * K.square(sigma2))
    #     KLD = first_term + second_term - 0.5
    #     return K.sum(K.sum(KLD, axis=-1), axis=-1)
    #
    # def variational_loss(y_true, y_pred):
    #     return (gaussian_log_likelihood(y_true, y_pred) +
    #             KL_divergence(Z_mean, Z_log_sigma, prior_mean, prior_log_sigma))

    filepath = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
    checkpoint_callback = SavePeriodicCheckpoint(
        filepath, monitor='val_loss', verbose=1, n_epochs=save_every)
    callbacks_list = [checkpoint_callback]

    train_gen = audio_amplitudes_gen(
        wavdir=wavdir, lyr_dir=lyr_dir, batch_size=batch_size,
        num_steps=num_steps, step_shift=step_shift, wav_dim=wav_dim)
    vae.fit_generator(
        train_gen,
        samples_per_epoch=batch_size*545, verbose=2,
        nb_epoch=num_epochs, callbacks=callbacks_list)

if __name__ == "__main__":
    args = parse_args(mode="train", use_phonemes=True)
    train(train_dir=args.train_dir, lyr_dir=args.lyr_dir,
          valid_dir=args.valid_dir, lyr_valid_dir=args.lyr_valid_dir,
          z_dim=args.z_dim, lstm_size=args.lstm_size, num_steps=args.num_steps,
          checkpoint_dir=args.checkpoint_dir, batch_size=args.batch_size,
          fc_dim=args.fc_dim, clip_grad=args.clip_grad, step_shift=args.step_shift,
          learning_rate=args.learning_rate, num_epochs=args.num_epochs,
          save_every=args.save_every, wav_dim=args.wav_dim)
