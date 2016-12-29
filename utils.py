import os
import numpy as np

from scipy.io.wavfile import read
from scipy.io.wavfile import write

from keras.preprocessing.sequence import pad_sequences


def build_phonemes_vocab(ph_dir):
    list_of_files = os.listdir(ph_dir)
    phonemes = []

    for file_ in list_of_files:
        curr_lines = open(os.path.join(ph_dir, file_), "r").readlines()
        curr_phs = set([f.strip().split(" ")[0] for f in curr_lines])
        phonemes.extend(list(curr_phs))

    phonemes = list(set(phonemes))
    ranges = np.arange(len(phonemes))
    return dict(zip(phonemes, ranges))


def return_time_phonemes(phonemes, phonemes_times, start_time, num_steps,
                         step_length, phoneme_dict, step_shift):
    """
    Returns phonemes of len(num_steps) with phonemes[i] corresponding
    to the number of phonemes at that time step.
    """
    start_step_times = (
        start_time + step_shift * step_length +
        step_length * np.arange(num_steps))
    end_step_times = np.array(
        list(start_step_times[1: ]) +
        [start_step_times[-1] + step_length])
    start_indices = np.searchsorted(phonemes_times, start_step_times) - 1
    end_indices = np.searchsorted(phonemes_times, end_step_times) - 1
    start_indices[start_indices == -1] = 0
    end_indices[end_indices == -1] = 0

    step_phonemes = np.zeros((num_steps, len(phoneme_dict)))
    ph_times = []
    for start_ind, end_ind in zip(start_indices, end_indices):
        ph_times.append(phonemes[start_ind: end_ind + 1])
    for i, step in enumerate(ph_times):
        for phoneme in step:
            step_phonemes[i, phoneme_dict[phoneme]] += 1
    return step_phonemes


def init_phonemes(phonemes, phonemes_times, start_time, num_steps,
                  step_length, phoneme_dict):
    """
    Returns phonemes of len(num_steps) with phonemes[i] corresponding
    to the number of phonemes at that time step.
    """
    start_time = start_time
    end_time = start_time + num_steps * step_length

    start_ind = np.searchsorted(phonemes_times, start_time) - 1
    if start_ind == -1:
        start_ind = 0
    end_ind = np.searchsorted(phonemes_times, end_time)

    # Encode all the phoneme information in the first step. Zero out
    # everything else as a hack to merge in only the first step.
    step_phonemes = np.zeros((num_steps, len(phoneme_dict)))

    for ph in phonemes[start_ind: end_ind]:
        step_phonemes[0, phoneme_dict[ph]] += 1
    return step_phonemes


def audio_amplitudes_gen(wavdir, phdir=None, batch_size=32, step_length=0.0125,
                         num_steps=40, random_state=None, step_shift=0):
    """
    Audio raw-amplitude batch generator.

    Step length - Predicts raw-amplitude of the next time-step using the
    given step.

    num_steps - Number of time steps of each sample.
    """
    rng = np.random.RandomState(random_state)
    limit_counter = 545

    if phdir is not None:
        phoneme_to_id = build_phonemes_vocab(phdir)

    wavfiles = os.listdir(wavdir)
    n_songs = len(wavfiles)
    xs = []
    ys = []
    startptr = 0
    song_ind = 0
    sample_size = int(step_length * num_steps * 16000)
    step_size = int(step_length * 16000)

    curr_wav = wavfiles[song_ind % n_songs]
    wavpath = os.path.join(wavdir, wavfiles[song_ind % n_songs])

    if phdir:
        lyrpath = os.path.join(phdir, curr_wav[:-4] + ".txt")
        curr_phonemes = open(lyrpath, "r").readlines()
        phonemes_times = [float(c.strip().split(" ")[-1]) for c in curr_phonemes]
        phonemes = [c.strip().split(" ")[0] for c in curr_phonemes]

    _, current_amps = read(wavpath)
    current_amps = current_amps / 32768.0

    start_time = 0.0
    counter = 0

    while True:
        counter += 1
        xs = []
        ys = []
        batch_phonemes = []
        batch_ind = 0

        while batch_ind < batch_size:
            if startptr + sample_size + step_size <= len(current_amps):
                x = np.reshape(
                    current_amps[startptr: startptr + sample_size], (num_steps, -1))
                y = np.reshape(
                    current_amps[startptr + step_size : startptr + sample_size + step_size],
                    (num_steps, -1))
                xs.append(x)
                ys.append(y)
                startptr += sample_size
                batch_ind += 1

                if phdir:
                    step_phonemes = return_time_phonemes(
                        phonemes, phonemes_times, start_time, num_steps, step_length,
                        phoneme_to_id, step_shift)
                    batch_phonemes.append(step_phonemes)
                start_time += num_steps * step_length
            else:
                song_ind += 1
                curr_wav = wavfiles[song_ind % n_songs]
                wavpath = os.path.join(wavdir, curr_wav)

                if phdir:
                    lyrpath = os.path.join(phdir, curr_wav[:-4] + ".txt")
                _, current_amps = read(wavpath)

                startptr = 0
                current_amps = current_amps / 32768.0

                if phdir:
                    curr_phonemes = open(lyrpath, "r").readlines()
                    phonemes_times = [float(c.strip().split(" ")[-1]) for c in curr_phonemes]
                    phonemes = [c.strip().split(" ")[0] for c in curr_phonemes]
                start_time = 0.0

        batch_ind = 0
        if counter % limit_counter == 0:
            rng.shuffle(wavfiles)
        xs = np.array(xs)
        ys = np.array(ys)

        if phdir:
            batch_phonemes = np.array(batch_phonemes)
            yield ([xs, ys, batch_phonemes], ys)
        else:
            yield ([xs, ys], ys)


def write_audio_utils(batch, dest_path):
    write(dest_path, 16000, (batch.ravel() * 32768).astype(np.int16))