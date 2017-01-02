# Eminet
Eminet is a variational LSTM for music generation. There are two versions:


1. The model that makes use of just the audio.
2. The model that takes in text input in addition to the raw waveforms. (The scripts that have `ph` in them)

<img src="https://cloud.githubusercontent.com/assets/1867024/21585903/5388b052-d0ef-11e6-8cae-036f633c2518.jpg", width=400, align="center">

## Dependencies

1. NumPy / SciPy

## How to run?

### Without text input
The model that takes just the audio is pretty straightforward to run.

```python
python3 vrnn_train.py --train_dir="path-to-dir-with-wav-files" --num_epochs=50 --save_every=10
```

This will save a keras model on disk prefixed with "weights-". You can then use this to predict and generate waveforms for new audio files.

```python
python3 vrnn_predict.py --wav_dir="path-to-dir-with-wav-files" --model="path-to-keras-model"

python3 vrnn_generate.py --wav_dir="path-direc-with-wav-files" --model="path-to-keras-model"
```

### With text input

1. Use 



## Sample output:



## Credits
