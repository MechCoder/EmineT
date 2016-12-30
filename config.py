import argparse

def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    if mode == "train":
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
            '--num_epochs', nargs="?", default=50, type=int,
            help="Number of epochs")
        parser.add_argument(
            '--save_every', nargs="?", default=5, type=int,
            help="Save the model every save_every number of epochs.")
    elif mode == "predict":
        parser.add_argument(
            '--wav_dir', nargs=1, type=str,
            help="Directory that contains wav files for prediction.")
        parser.add_argument(
            '--model', nargs=1, type=str, help="Path to the trained model")
        parser.add_argument(
            '--pred_dir', nargs="?", default="wav_dir", type=str,
            help="Directory that the predicted wav files will be written to.")

    parser.add_argument(
        '--lstm_size', nargs="?", default=1000, type=int,
        help="Number of hidden lstm units.")
    parser.add_argument(
        '--num_steps', nargs="?", default=40, type=int,
        help="Number of lstm time-steps.")
    parser.add_argument(
        '--wav_dim', nargs="?", default=200, type=int,
        help="Dimension of the input waveform.")
    parser.add_argument('--z_dim', nargs="?", default=100, type=int,
        help="Latent size dimensions.")
    parser.add_argument('--batch_size', nargs="?", default=32, type=int,
        help="Batch size.")
    parser.add_argument('--fc_dim', nargs="?", default=400, type=int,
        help="Dimension of the input fully-connected layer before providing as "
        "input to LSTM")
    parser.add_argument('--learning_rate', nargs="?", default=0.001, type=float,
        help="Learning rate of the Adam optimizer.")
    parser.add_argument('--clip_grad', nargs="?", default=5.0, type=float,
        help="Clip the value of gradients above clip_grad to clip_grad")
    return parser.parse_args()
