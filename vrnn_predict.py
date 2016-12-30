import os
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

from config import parse_args
from math import pi
from utils import audio_amplitudes_gen
from utils import write_audio

def predict(wav_dir, model, write_dir, lstm_size=1000, num_steps=40,
            z_dim=100, batch_size=32, fc_dim=400, wav_dim=200,
            learning_rate=0.001, clip_grad=5.0):
    input_ = Input(batch_shape=(batch_size, num_steps, wav_dim))
    # Input but shifed by one-time step
    input_shift = Input(batch_shape=(batch_size, num_steps, wav_dim))
    higher = TimeDistributed(Dense(fc_dim, activation="tanh"))(input_)

    # Vanilla LSTM
    hidden = LSTM(lstm_size, return_sequences=True)(higher)

    # Prior on the latent variables (z_{t + 1}) is Dependent on the input
    prior_mean = TimeDistributed(Dense(z_dim, activation="tanh"))(hidden)
    prior_log_sigma = TimeDistributed(Dense(z_dim, activation="relu"))(hidden)

    # Merge hidden-state and input to form the encoder network.
    hidden_to_z = TimeDistributed(Dense(z_dim, activation="relu"))(hidden)
    input_to_z = TimeDistributed(Dense(z_dim, activation="relu"))(input_shift)
    hidden_with_input = merge([hidden_to_z, input_to_z], mode="sum")
    Z_mean = TimeDistributed(Dense(z_dim, activation="tanh"))(hidden_with_input)
    Z_log_sigma = TimeDistributed(Dense(z_dim, activation="relu"))(hidden_with_input)

    def sampling(args):
        Z_mean, Z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, num_steps, z_dim))
        return Z_mean + K.exp(Z_log_sigma) * epsilon

    samples = Lambda(sampling)([Z_mean, Z_log_sigma])
    hidden_to_out = TimeDistributed(Dense(wav_dim))(hidden)
    samples_to_out = TimeDistributed(Dense(wav_dim))(samples)
    hid_merge_samples = merge([hidden_to_out, samples_to_out], mode="sum")
    out_mu = TimeDistributed(Dense(wav_dim, activation="tanh"))(hid_merge_samples)
    out_log_sigma = TimeDistributed(Dense(wav_dim, activation="relu"))(hid_merge_samples)

    def gaussian_log_likelihood(y_true, y_pred):
        sigma = K.exp(out_log_sigma)
        term1 = 0.5 * K.mean(K.square((y_true - y_pred) / sigma), axis=-1)
        term2 = K.mean(out_log_sigma, axis=-1)
        term3 = 0.5 * K.log(2 * pi)
        return K.sum(term1 + term2 + term3, axis=-1)

    def KL_divergence(mu1, logsigma1, mu2, logsigma2):
        sigma1 = K.exp(logsigma1)
        sigma2 = K.exp(logsigma2)
        first_term = logsigma2 - logsigma1
        second_term = (K.square(sigma1) + K.square(mu1 - mu2)) / (2 * K.square(sigma2))
        KLD = first_term + second_term - 0.5
        return K.sum(K.mean(KLD, axis=-1), axis=-1)

    def variational_loss(y_true, y_pred):
        return (gaussian_log_likelihood(y_true, y_pred) +
                KL_divergence(Z_mean, Z_log_sigma, prior_mean, prior_log_sigma))

    adam = Adam(lr=learning_rate, clipnorm=clip_grad)
    vae = Model(input=[input_, input_shift], output=out_mu)
    vae.load_weights(model)

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    counter = 0
    pred_gen = audio_amplitudes_gen(
        wavdir=wav_dir, num_steps=num_steps, batch_size=batch_size,
        wav_dim=wav_dim, infinite=False)

    for (x_t, y_t), true in pred_gen:
        pred = vae.predict([x_t, y_t], batch_size=batch_size)
        print("Writing audio %d" % counter)
        true_path = os.path.join(write_dir, "%d_true.wav" % counter)
        pred_path = os.path.join(write_dir, "%d_pred.wav" % counter)
        write_audio(true, true_path)
        write_audio(pred, pred_path)
        counter += 1


if __name__ == "__main__":
    args = parse_args(mode="predict")
    predict(args.wav_dir, args.model, args.write_dir,
            z_dim=args.z_dim, lstm_size=args.lstm_size, num_steps=args.num_steps,
            batch_size=args.batch_size,
            fc_dim=args.fc_dim, clip_grad=args.clip_grad,
            learning_rate=args.learning_rate, wav_dim=args.wav_dim)
