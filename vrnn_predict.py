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

from math import pi
from new_audio_utils import gen_audio_phonemes_pairs
from new_audio_utils import write_audio_utils
# Hyper-Parameters
hidden_size = 1000
inter_dim = 400
z_dim = 100
batch_size = 32
num_steps = 40
in_dim = 200
step_shift = 0

checkpoint_dir = os.path.join(os.getcwd(), "vrnn_checkpoints")
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

input_ = Input(batch_shape=(batch_size, num_steps, in_dim))

# Input but shifed by one-time step
input_shift = Input(batch_shape=(batch_size, num_steps, in_dim))
higher = TimeDistributed(Dense(inter_dim, activation="tanh"))(input_)

# Vanilla LSTM
hidden = LSTM(hidden_size, return_sequences=True)(higher)

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
hidden_to_out = TimeDistributed(Dense(in_dim))(hidden)
samples_to_out = TimeDistributed(Dense(in_dim))(samples)
hid_merge_samples = merge([hidden_to_out, samples_to_out], mode="sum")
out_mu = TimeDistributed(Dense(in_dim, activation="tanh"))(hid_merge_samples)
out_log_sigma = TimeDistributed(Dense(in_dim, activation="relu"))(hid_merge_samples)

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


adam = Adam(lr=0.001, clipnorm=5.0)
vae = Model(input=[input_, input_shift], output=out_mu)
encoder = Model(input=[input_, input_shift], output=Z_mean)

checkpoint_files = sorted(os.listdir(checkpoint_dir))
if len(checkpoint_files) != 0:

    for epoch_ind, checkpoint in enumerate(checkpoint_files):
        print("Loading weights")
        curr_ckpt = os.path.join(checkpoint_dir, checkpoint)
        vae.load_weights(curr_ckpt)

        epoch_dir = os.path.join(checkpoint_dir, str(epoch_ind))
        os.mkdir(epoch_dir)

        counter = 0
        for (x_t, y_t), true in gen_audio_phonemes_pairs(return_y=True, path="valid"):

            pred = vae.predict([x_t, y_t], batch_size=32)
            counter += 1

            print("Writing audio %d" % counter)
            true_path = os.path.join(epoch_dir, "%d_true.wav" % counter)
            pred_path = os.path.join(epoch_dir, "%d_pred.wav" % counter)
            write_audio_utils(true, true_path)
            write_audio_utils(pred, pred_path)

            if counter == 20:
                break
