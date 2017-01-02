import json
import numpy as np
import os
import shutil
import sys

from scipy.io.wavfile import read
from scipy.io.wavfile import write


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


def samples_per_epoch(wavdir, batch_size=32, num_steps=40, wav_dim=200):
    wavfiles = os.listdir(wavdir)
    batch_dim = wav_dim * num_steps * batch_size
    n_samples = 0
    remainder = 0
    for wavfile in wavfiles:
        wavpath = os.path.join(wavdir, wavfile)
        _, amps = read(wavpath)
        n_samples += int(len(amps) / batch_dim)
        remainder += len(amps) % batch_dim
    return n_samples + int(remainder / batch_dim)


def split_words(lyrics_dir, num_words=3):
    txt_files = os.listdir(lyrics_dir)
    for txt_file in txt_files:
        lines = open(os.path.join(lyrics_dir, txt_file), "r")
        lyrics = lines.readlines()
        lines.close()

        split_lines = open(os.path.join(lyrics_dir, txt_file[:-4]) + "_split.txt", "w")
        words = [word for l in lyrics for word in l.strip().split(" ")]
        for i in range(len(words) // num_words):
            per_lines = [words[j + i * num_words]  for j in range(num_words)]
            split_lines.write(" ".join(per_lines) + "\n")

        startptr =  len(words) // num_words * num_words
        rem = len(words) - startptr
        per_lines = [words[j + startptr] for j in range(rem)]
        split_lines.write(" ".join(per_lines) + "\n")
        split_lines.close()


def synchronize(song_dir, lyrics_dir, out_dir="time_stamp"):
    """
    Synchronize songs with lyrics and outputs a directory with files
    containing the start time of each phoneme.
    """
    if os.path.exists("json_dir"):
        shutil.rmtree("json_dir")
    os.mkdir("json_dir")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    songs = os.listdir(song_dir)
    for song in songs:
        song_path = os.path.join(song_dir, song)
        lyric = os.path.join(lyrics_dir, song[:-4] + ".txt")
        json_path = os.path.join("json_dir", song[:-4] + ".json")
        time_path = os.path.join(out_dir, song[:-4] + ".txt")

        os.system("python%d -m aeneas.tools.execute_task \
                  %s %s 'task_language=eng|os_task_file_format=json|is_text_type=plain' %s" %
                  (sys.version_info[0], song_path, lyric, json_path))

        lines = []
        json_file = open(json_path, "r")
        for time_stamp in json.load(json_file)['fragments']:
            end_time = float(time_stamp["end"])
            start_time = float(time_stamp["begin"])
            phonemes = str(time_stamp['lines'][0]).split(" ")
            interval = (end_time - start_time) / len(phonemes)
            times = np.arange(len(phonemes)) * interval + start_time
            for ph, st_time in zip(phonemes, times):
                lines.append(ph + " " + str(st_time) + "\n")

        b = open(time_path, "w")
        b.writelines(lines)
        b.close()

    shutil.rmtree("json_dir")


def audio_amplitudes_gen(wavdir, lyr_dir=None, batch_size=32,
                         num_steps=40, random_state=None, step_shift=0,
                         wav_dim=200, infinite=True):
    """
    Audio raw-amplitude batch generator.

    Step length - Predicts raw-amplitude of the next time-step using the
    given step.

    num_steps - Number of time steps of each sample.
    """
    rng = np.random.RandomState(random_state)
    n_samples = samples_per_epoch(wavdir, batch_size, num_steps, wav_dim)

    if lyr_dir is not None:
        phoneme_to_id = build_phonemes_vocab(lyr_dir)

    wavfiles = os.listdir(wavdir)
    n_songs = len(wavfiles)
    xs = []
    ys = []
    startptr = 0
    song_ind = 0
    step_length = wav_dim / 16000.0
    sample_size = int(wav_dim * num_steps)

    curr_wav = wavfiles[song_ind % n_songs]
    wavpath = os.path.join(wavdir, wavfiles[song_ind % n_songs])

    if lyr_dir:
        lyrpath = os.path.join(lyr_dir, curr_wav[:-4] + ".txt")
        curr_phonemes = open(lyrpath, "r").readlines()
        phonemes_times = [float(c.strip().split(" ")[-1]) for c in curr_phonemes]
        phonemes = [c.strip().split(" ")[0] for c in curr_phonemes]

    _, current_amps = read(wavpath)
    current_amps = current_amps / 32768.0

    start_time = 0.0
    counter = 0

    # XXX: Ugly, but I'll change it some other day.
    while True:
        counter += 1
        xs = []
        ys = []
        batch_phonemes = []
        batch_ind = 0

        while batch_ind < batch_size:
            if startptr + sample_size + wav_dim <= len(current_amps):
                x = np.reshape(
                    current_amps[startptr: startptr + sample_size], (num_steps, -1))
                y = np.reshape(
                    current_amps[startptr + wav_dim: startptr + sample_size + wav_dim],
                    (num_steps, -1))
                xs.append(x)
                ys.append(y)
                startptr += sample_size
                batch_ind += 1

                if lyr_dir:
                    step_phonemes = return_time_phonemes(
                        phonemes, phonemes_times, start_time, num_steps, step_length,
                        phoneme_to_id, step_shift)
                    batch_phonemes.append(step_phonemes)
                start_time += num_steps * step_length
            else:
                song_ind += 1
                curr_wav = wavfiles[song_ind % n_songs]
                wavpath = os.path.join(wavdir, curr_wav)

                if lyr_dir:
                    lyrpath = os.path.join(lyr_dir, curr_wav[:-4] + ".txt")
                _, current_amps = read(wavpath)

                startptr = 0
                current_amps = current_amps / 32768.0

                if lyr_dir:
                    curr_phonemes = open(lyrpath, "r").readlines()
                    phonemes_times = [float(c.strip().split(" ")[-1]) for c in curr_phonemes]
                    phonemes = [c.strip().split(" ")[0] for c in curr_phonemes]
                start_time = 0.0

        batch_ind = 0
        if song_ind % n_songs == 0 and song_ind != 0:
            if infinite:
                rng.shuffle(wavfiles)
            else:
                return
        xs = np.array(xs)
        ys = np.array(ys)

        if lyr_dir:
            batch_phonemes = np.array(batch_phonemes)
            yield ([xs, ys, batch_phonemes], ys)
        else:
            yield ([xs, ys], ys)


def write_audio(batch, dest_path):
    write(dest_path, 16000, (batch.ravel() * 32768).astype(np.int16))
