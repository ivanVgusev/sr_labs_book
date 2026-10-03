import numpy as np
from scipy.fftpack import dct


def split_meta_line(line, delimiter=' '):
    if delimiter == ' ':
        parts = line.strip().split()
    else:
        parts = line.strip().split(delimiter)

    if len(parts) != 3:
        raise ValueError('Строка метаданных должна содержать три значения')

    speaker_id, gender, file_path = parts
    return speaker_id, gender, file_path


def preemphasis(signal, pre_emphasis=0.97):
    signal = np.asarray(signal)
    if signal.size == 0:
        return signal.copy()

    emphasized_signal = np.empty(signal.shape, dtype=float)
    emphasized_signal[0] = signal[0]
    emphasized_signal[1:] = signal[1:] - pre_emphasis * signal[:-1]

    return emphasized_signal


def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    signal = np.asarray(emphasized_signal)
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))

    if frame_length <= 0 or frame_step <= 0:
        raise ValueError('Размер фрейма и шаг должны быть больше нуля')

    signal_length = len(signal)
    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_length - frame_length) / frame_step))

    padded_length = (num_frames - 1) * frame_step + frame_length
    padded_signal = np.zeros(padded_length, dtype=float)
    padded_signal[:signal_length] = signal

    frames = np.zeros((num_frames, frame_length), dtype=float)
    for frame_number in range(num_frames):
        start = frame_number * frame_step
        end = start + frame_length
        frames[frame_number] = padded_signal[start:end]

    frames *= np.hamming(frame_length)
    return frames


def power_spectrum(frames, NFFT=512):
    if NFFT <= 0:
        raise ValueError('NFFT должен быть больше нуля')

    magnitude_spectrum = np.abs(np.fft.rfft(frames, NFFT))
    pow_frames = magnitude_spectrum ** 2 / NFFT

    return pow_frames


def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    if nfilt <= 0 or sample_rate <= 0 or NFFT <= 0:
        raise ValueError('Параметры банка фильтров должны быть больше нуля')

    low_frequency_mel = 0
    high_frequency = sample_rate / 2
    high_frequency_mel = 2595 * np.log10(1 + high_frequency / 700)

    mel_points = np.linspace(low_frequency_mel, high_frequency_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, NFFT // 2)

    fbank = np.zeros((nfilt, NFFT // 2 + 1))

    for filter_number in range(1, nfilt + 1):
        left = bins[filter_number - 1]
        center = bins[filter_number]
        right = bins[filter_number + 1]

        if center > left:
            for frequency_bin in range(left, center):
                fbank[filter_number - 1, frequency_bin] = (
                    (frequency_bin - left) / (center - left)
                )

        if right > center:
            for frequency_bin in range(center, right):
                fbank[filter_number - 1, frequency_bin] = (
                    (right - frequency_bin) / (right - center)
                )

    return fbank


def compute_fbanks_features(pow_frames, fbank):
    filter_banks_features = np.dot(pow_frames, fbank.T)
    filter_banks_features = np.maximum(
        filter_banks_features,
        np.finfo(float).eps,
    )
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features


# Было: num_ceps=20
def compute_mfcc(filter_banks_features, num_ceps=23):
    if num_ceps <= 0:
        raise ValueError('Количество MFCC должно быть больше нуля')

    all_coefficients = dct(filter_banks_features, type=2, axis=1, norm='ortho')
    mfcc = all_coefficients[:, 1:num_ceps + 1]

    return mfcc


def mvn_floating(features, LC, RC, unbiased=False):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError('Признаки должны быть двумерной матрицей')
    if LC < 0 or RC < 0:
        raise ValueError('Размеры левого и правого окна не могут быть отрицательными')
    if len(features) == 0:
        return features.copy()

    normalised_features = np.zeros_like(features)
    correction = 1 if unbiased else 0

    for frame_number in range(len(features)):
        left = max(0, frame_number - LC)
        right = min(len(features), frame_number + RC + 1)
        window = features[left:right]

        mean = np.mean(window, axis=0)
        if len(window) <= correction:
            standard_deviation = np.zeros(features.shape[1])
        else:
            standard_deviation = np.std(window, axis=0, ddof=correction)

        nonzero = standard_deviation > 0
        normalised_features[frame_number, nonzero] = (
            features[frame_number, nonzero] - mean[nonzero]
        ) / standard_deviation[nonzero]

    return normalised_features
