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
            z_dim=100, batch_size=32, fc_dim=400, wav_dim=200):

    vae = build_vrnn(lstm_size=lstm_size, num_steps=num_steps, z_dim=z_dim,
                     batch_size=batch_size, fc_dim=fc_dim, wav_dim=wav_dim)
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


if __name__ == "__main__":
    args = parse_args(mode="predict")
    predict(args.wav_dir, args.model, args.write_dir,
            z_dim=args.z_dim, lstm_size=args.lstm_size, num_steps=args.num_steps,
            batch_size=args.batch_size,
            fc_dim=args.fc_dim, wav_dim=args.wav_dim)
