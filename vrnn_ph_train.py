import os
from keras.layers import Input, Lambda, LSTM, Dense, TimeDistributed, merge, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from callbacks import SavePeriodicCheckpoint

from math import pi
from utils import gen_audio_phonemes_pairs

# Hyper-Parameters
hidden_size = 1000
z_dim = 100
batch_size = 32
num_steps = 40
in_dim = 200
phoneme_length = 40
phonemes_embed_size = 400
step_shift = 0

checkpoint_dir = os.path.join(os.getcwd(), "vrnn_lyrics_checkpoints")
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

input_ = Input(batch_shape=(batch_size, num_steps, in_dim))
input_shift = Input(batch_shape=(batch_size, num_steps, in_dim))
ph_input = Input(batch_shape=(batch_size, num_steps, phoneme_length))
counts = K.expand_dims(K.sum(ph_input, axis=-1), -1)

# Merge input from phonemes and audio.
cbow = TimeDistributed(Dense(phonemes_embed_size))(ph_input)
cbow_averaged = Lambda(lambda x: x / counts)(cbow)
input_layer2 = TimeDistributed(Dense(phonemes_embed_size, activation="relu"))(input_)
lyrics_plus_audio = merge([input_layer2, cbow_averaged], mode="sum")

# Vanilla LSTM
hidden = LSTM(hidden_size, return_sequences=True)(lyrics_plus_audio)

# Prior on the latent variables (z_{t + 1}) is Dependent on the input
# hidden_firs = TimeDistributed(Dense(z_dim, activation="relu"))(lstm_hidden)
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

filepath = os.path.join(checkpoint_dir, "{epoch:02d}.hdf5")
checkpoint_callback = SavePeriodicCheckpoint(
    filepath, monitor='val_loss', verbose=1, n_epochs=5)
callbacks_list = [checkpoint_callback]

adam = Adam(lr=0.001, clipnorm=5.0)
vae = Model(input=[input_, input_shift, ph_input], output=out_mu)
encoder = Model(input=[input_, input_shift, ph_input], output=Z_mean)

vae.compile(optimizer=adam, loss=variational_loss)
checkpoint_files = sorted(os.listdir("vrnn_lyrics_checkpoints"))
if len(checkpoint_files) != 0:
    print("Loading weights")
    # Prevents the previous checkpoints from getting over-written
    for c in checkpoint_files:
        curr_ckpt = os.path.join(checkpoint_dir, c)
        new_name = os.path.join(checkpoint_dir, "prev_" + c)
        os.rename(curr_ckpt, new_name)
    vae.load_weights(new_name)


vae.fit_generator(
    gen_audio_phonemes_pairs(return_phonemes=True, step_shift=step_shift),
    samples_per_epoch=batch_size*545, verbose=2,
    nb_epoch=50,
    validation_data=gen_audio_phonemes_pairs(
        return_phonemes=True, path="valid", step_shift=step_shift),
    nb_val_samples=batch_size*147, callbacks=callbacks_list)
