# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal
from collections import defaultdict

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    def seconds_to_samples(seconds: str):
        return int(fs * float(seconds))
    
    with open(path_to_rttm) as f:
        rttm = f.readlines()
        
    rttm_stamps = []
    for line in rttm:
        info = line.split(" ")
        channel = info[2]
        voice_start = seconds_to_samples(info[3])
        voice_end = voice_start + seconds_to_samples(info[4])

        rttm_stamps.append([
            channel,
            voice_start, 
            voice_end
        ])
    vad_markup = np.zeros(len(signal)).astype('float32')
    for interval in rttm_stamps:
        vad_markup[interval[1]:interval[2]] = np.ones(
            interval[2] - interval[1]
        ).astype('float32')

    return vad_markup


def framing(signal, window, shift):
    signal = np.asarray(signal)

    if window <= 0 or shift <= 0:
        raise ValueError('Размер окна и шаг должны быть больше нуля')

    if len(signal) <= window:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((len(signal) - window) / shift))

    padded_length = (num_frames - 1) * shift + window
    padded_signal = np.zeros(padded_length)
    padded_signal[:len(signal)] = signal

    frames = np.zeros((num_frames, window))
    for i in range(num_frames):
        start = i * shift
        frames[i] = padded_signal[start:start + window]

    return frames


def frame_energy(frames):
    frame_energies = np.sum(np.abs(frames) ** 2, axis=1)
    return frame_energies


def norm_energy(E):
    return (E - np.mean(E)) / np.std(E)


def gmm_train(E, gauss_pdf, n_realignment):
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):
        # E-step
        p = []
        for n_gaussian in range(len(w)):
            p.append(gauss_pdf(E, m[n_gaussian], sigma[n_gaussian]))
        p = np.array(p).T
        q = p * w
        g = q / np.sum(q, axis=1, keepdims=True)

        # M-step
        N_k = np.sum(g, axis=0)
        w = N_k / len(g)
        m = (g * E[:, None]).sum(axis=0) / N_k
        sigma = np.sqrt((g * (E[:, None] - m)**2).sum(axis=0) / N_k)
        
    return w, m, sigma


def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    p = []
    for n_gaussian in range(len(w)):
        p.append(gauss_pdf(E, m[n_gaussian], sigma[n_gaussian]))
    p = np.array(p).T
    q = p * w
    g = q / np.sum(q, axis=1, keepdims=True)

    silence_idx = np.argmin(m)
    g0 = g[:, silence_idx]
    return g0


def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Frame the original signal with overlap.  frame_energy applies the
    # square while calculating the energy of each frame.
    frames = framing(signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real


def reverb(signal, impulse_response):
    signal_reverb = scipy.signal.fftconvolve(
        signal,
        impulse_response,
        mode="full"
    )

    signal_reverb = signal_reverb[:len(signal)]

    return signal_reverb.astype("float32")


def awgn(signal, sigma_noise):
    m = 0
    noise = np.random.normal(m, sigma_noise, len(signal))
    signal_noise = np.clip(signal + noise, -1, 1)

    return signal_noise
