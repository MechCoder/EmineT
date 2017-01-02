# Eminet
Eminet is a variational LSTM for rap generation. There are two versions:


1. The model that makes use of just the audio.
2. The model that takes in text input in addition to the raw waveforms. (The scripts that have `ph` in them)

<img src="https://cloud.githubusercontent.com/assets/1867024/21585903/5388b052-d0ef-11e6-8cae-036f633c2518.jpg", width=400, align="center">

## Dependencies

1. FFMpeg
2. NumPy
3. SciPy
4. Keras
5. Aeneas (For lyrics synchronization)

## How to run?

### Without text input
The model that takes just the audio is pretty straightforward to run. It is assumed that the input wav files are converted from stereo to mono with a sampling frequency of 16kHZ.

```python
python3 vrnn_train.py --train_dir="path-to-dir-with-wav-files" --num_epochs=50 --save_every=10
```

This will save a keras model on disk prefixed with "weights-". You can then use this to predict and generate waveforms for new audio files.

```python
python3 vrnn_predict.py --wav_dir="path-to-dir-with-wav-files" --model="path-to-keras-model"

python3 vrnn_generate.py --wav_dir="path-direc-with-wav-files" --model="path-to-keras-model"
```

### With text input

The preprocessing is slightly more complicated.

* Restructure the text file containing lyrics such that there are a sufficient number of words per line. This is done so that aeneas can synchronize effectively.

```python
from utils import split_words
split_words(lyrics_dir, num_words)
```
For this project, we limit the number of words to 3. This will generate files with ``_split`` in them.

* Use the online CMU phoneme converter (http://www.speech.cs.cmu.edu/tools/lextool.html) to convert the words into phonemes.  This will generate output similar to this.

```
Y'ALL KNOW ME,	Y AO L N OW M IY
STILL THE SAME	S T IH L DH AH S EY M
```

This is tab-separated, so it is easy to separate the phonemes.

* Synchronize the phonemes to the wav files and get the start times.

```python
# Here ph_dir is the directory that contains the phonemes as obtained from the prev
# step. The file containing the phonemes corresposnding to song1.wav should be named as
# song1.txt
from utils import synchronize
synchronize(song_dir, ph_dir)
```

This will generate a directory `time_stamp` that contains the start time for every phoneme.

* Now with this additional information, we can run it as before

```python
python3 vrnn_ph_train.py --train_dir="path-to-dir-with-wav-files" --lyr_dir="path_to_time_stamp' "--num_epochs=50 --save_every=10
```
(You can use the `--help` flag for the documentation of hyper-parameters.)


## Sample output:

The sample output for the predicted waveforms on unseen data at the end of the 5th and 50th epoch can be listened to here. The Ph prefix refers to the model that takes phonemes as input. The model was trained on 40 rap songs of Eminem (Hence the name)

* https://soundcloud.com/manoj-kumar-475/sets/sample-1-vrnn
* https://soundcloud.com/manoj-kumar-475/sets/sample2

The sample output for the generated waveforms at the end of the 5th and 50th epoch can be listened to here.

* https://soundcloud.com/manoj-kumar-475/sets/gen_1
* https://soundcloud.com/manoj-kumar-275045882/sets/gen2

## Credits

This project was done in partial-fulfillment of DS-GA 1007 (Inference and Representation). Thanks to Prof. David Sontag and the amazing open-source tools that we used on the way.
