from keras.callbacks import ModelCheckpoint

class SavePeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 n_epochs=5, save_weights_only=False,
                 mode='auto'):
        super(SavePeriodicCheckpoint, self).__init__(
            filepath=filepath, monitor=monitor, verbose=verbose,
            save_weights_only=save_weights_only,
            mode=mode, save_best_only=False)
        self.n_epochs = n_epochs
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch += 1
        if self.current_epoch % self.n_epochs == 0:
            super(SavePeriodicCheckpoint, self).on_epoch_end(
                epoch=epoch, logs=logs)
        else:
            pass
