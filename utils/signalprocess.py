import os, sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)
import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal


def samples_to_stft_frames(samples, size, shift, ceil=False):
    if ceil:
        return 1 if samples <= size - shift else \
                np.ceil((samples - size + shift) / shift).astype(np.int32)
    else:
        return 1 if samples <= size else (samples - size + shift) // shift

def stft_frames_to_samples(frames, size, shift):
    return frames * shift + size - shift


# compute stft of a 1-dim time_signal
def stft(time_signal, size=1024, shift=256, fading=True, ceil=False,
         window=signal.windows.hann, window_length=None):
    assert time_signal.ndim == 1
    if fading:
        pad = [(size - shift, size - shift)]
        time_signal = np.pad(time_signal, pad, mode='constant')
    frames = samples_to_stft_frames(time_signal.shape[0], size,
                                    shift, ceil=ceil)
    samples = stft_frames_to_samples(frames, size, shift)
    if samples > time_signal.shape[0]:
        pad = [(0, samples - time_signal.shape[0])]
        time_signal = np.pad(time_signal, pad, mode='constant')
    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')
    chunk_signal = np.zeros((frames, size))
    for i, j in enumerate(range(0, samples - size + shift, shift)):
        chunk_signal[i] = time_signal[j:j+size]
    return rfft(chunk_signal * window, axis=1)


def istft(stft_signal, size=1024, shift=256, fading=True,
          window=signal.windows.hann, window_length=None):
    assert stft_signal.shape[1] == size // 2 + 1
    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')
    time_signal = np.zeros(stft_signal.shape[0] * shift + size - shift)
    w = np.zeros(time_signal.shape)
    for i, j in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[j:j+size] += window * np.real(irfft(stft_signal[i], size))
        w[j:j+size] = window ** 2
    pos = (w != 0)
    time_signal[pos] /= w[pos]
    if fading:
        time_signal = time_signal[size - shift:len(time_signal) - size + shift]
    return time_signal.astype(np.float32)
