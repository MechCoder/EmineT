import os
from keras.layers import Input, Lambda, LSTM, Dense, TimeDistributed, merge, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from callbacks import SavePeriodicCheckpoint

from math import pi
from config import parse_args
from utils import audio_amplitudes_gen
from utils import samples_per_epoch
from vrnn_model import build_vrnn

def train(train_dir, lyr_dir, valid_dir=None, lyr_valid_dir=None,
          lstm_size=1000, num_steps=40, step_shift=0,
          z_dim=100, batch_size=32, fc_dim=400, wav_dim=200,
          checkpoint_dir="vrnn_checkpoints", learning_rate=0.001, clip_grad=5.0,
          num_epochs=50, save_every=5):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    vae = build_vrnn(lstm_size=lstm_size, num_steps=num_steps, z_dim=z_dim,
                     batch_size=batch_size, fc_dim=fc_dim, wav_dim=wav_dim,
                     learning_rate=learning_rate, clip_grad=clip_grad,
                     use_phonemes=True)

    filepath = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
    checkpoint_callback = SavePeriodicCheckpoint(
        filepath, monitor='val_loss', verbose=1, n_epochs=save_every)
    callbacks_list = [checkpoint_callback]

    train_gen = audio_amplitudes_gen(
        wavdir=train_dir, lyr_dir=lyr_dir, batch_size=batch_size,
        num_steps=num_steps, step_shift=step_shift, wav_dim=wav_dim)
    n_train_per_epoch = samples_per_epoch(
        wavdir=train_dir, batch_size=batch_size, num_steps=num_steps,
        wav_dim=wav_dim)

    if valid_dir is not None:
        valid_gen = audio_amplitudes_gen(
            wavdir=valid_dir, lyr_dir=lyr_valid_dir, num_steps=num_steps,
            batch_size=batch_size, wav_dim=wav_dim, step_shift=step_shift)
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
            train_gen,
            samples_per_epoch=batch_size*n_train_per_epoch, verbose=2,
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
