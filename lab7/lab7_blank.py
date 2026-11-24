#!/usr/bin/env python

import os
import random
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
ROOT_DIR = LAB_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = LAB_DIR / 'synth_audio_example'
sys.path.insert(0, str(LAB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    SpectralClustering,
)

from common.voxceleb_paths import VOX1_TEST_WAV, require_voxceleb
from lab3.ResNetBlocks import BasicBlock
from lab3.exercises_blank import ResNet


SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
STEP_SEC = 0.5

CLUSTER_ALGO = 'agglo'
N_SPEAKERS = 3
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 5

SYNTH_OUTPUT = OUTPUT_DIR / 'synth_audio.wav'
TURN_SEC = 3.0
N_TURNS_PER_SPK = 3


def list_voxceleb_utts(base_path):
    spk2utts = {}
    for spk in os.listdir(base_path):
        spk_dir = os.path.join(base_path, spk)
        if not os.path.isdir(spk_dir):
            continue

        files = []
        for root, _, fnames in os.walk(spk_dir):
            for filename in fnames:
                if filename.endswith('.wav'):
                    files.append(os.path.join(root, filename))

        if files:
            spk2utts[spk] = files

    return spk2utts


def run_clustering(embeddings):
    if CLUSTER_ALGO == 'kmeans':
        model = KMeans(n_clusters=N_SPEAKERS, random_state=0)
    elif CLUSTER_ALGO == 'agglo':
        try:
            model = AgglomerativeClustering(
                n_clusters=N_SPEAKERS,
                linkage='average',
                metric='cosine',
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=N_SPEAKERS,
                linkage='average',
                affinity='cosine',
            )
    elif CLUSTER_ALGO == 'spectral':
        model = SpectralClustering(
            n_clusters=N_SPEAKERS,
            assign_labels='kmeans',
            random_state=0,
            affinity='nearest_neighbors',
        )
    elif CLUSTER_ALGO == 'dbscan':
        model = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric='cosine',
        )
    else:
        raise ValueError('Unknown CLUSTER_ALGO')

    return model.fit_predict(embeddings)


def merge_labels_to_segments(labels, starts, win_len, sample_rate):
    if len(labels) == 0:
        return []

    segments = []
    current_label = int(labels[0])
    current_start = starts[0]
    for index in range(1, len(labels)):
        if int(labels[index]) != current_label:
            segments.append((
                current_start / sample_rate,
                (starts[index - 1] + win_len) / sample_rate,
                current_label,
            ))
            current_label = int(labels[index])
            current_start = starts[index]

    segments.append((
        current_start / sample_rate,
        (starts[-1] + win_len) / sample_rate,
        current_label,
    ))

    for index in range(len(segments) - 1):
        border = (segments[index][1] + segments[index + 1][0]) / 2
        segments[index] = (
            segments[index][0],
            border,
            segments[index][2],
        )
        segments[index + 1] = (
            border,
            segments[index + 1][1],
            segments[index + 1][2],
        )

    return segments


require_voxceleb(VOX1_TEST_WAV)
print(f'VoxCeleb1 test: {VOX1_TEST_WAV}')

random.seed(42)
spk2utts = list_voxceleb_utts(VOX1_TEST_WAV)
assert len(spk2utts) >= N_SPEAKERS, 'Not enough speakers in VoxCeleb1 test set.'

chosen_spks = random.sample(list(spk2utts), N_SPEAKERS)
turns = []
turn_labels = []
silence = torch.zeros(int(0.2 * SAMPLE_RATE))

for _ in range(N_TURNS_PER_SPK):
    for label, speaker in enumerate(chosen_spks):
        utterance_path = random.choice(spk2utts[speaker])
        wav, sample_rate = torchaudio.load(utterance_path)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)

        wav = wav.mean(dim=0)
        length = int(TURN_SEC * SAMPLE_RATE)
        if wav.numel() < length:
            repeats = int(np.ceil(length / wav.numel()))
            wav = wav.tile(repeats)[:length]
        else:
            start = random.randint(0, wav.numel() - length)
            wav = wav[start:start + length]

        turns.extend((wav, silence))
        turn_labels.extend((label, -1))

mixture = torch.cat(turns)
position = 0.0
segments_ref = []
for segment, label in zip(turns, turn_labels):
    duration = segment.numel() / SAMPLE_RATE
    if label >= 0:
        segments_ref.append((position, position + duration, label))
    position += duration

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torchaudio.save(SYNTH_OUTPUT, mixture.unsqueeze(0), SAMPLE_RATE)

ref_rttm = SYNTH_OUTPUT.with_name(SYNTH_OUTPUT.stem + '_ref.rttm')
recording_id = SYNTH_OUTPUT.stem
ref_lines = []
for start, end, label in segments_ref:
    duration = max(0.0, end - start)
    ref_lines.append(
        f'SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} '
        f'<NA> <NA> speaker_{label} <NA> <NA>'
    )
ref_rttm.write_text('\n'.join(ref_lines))

waveform, sample_rate = torchaudio.load(SYNTH_OUTPUT)
assert sample_rate == SAMPLE_RATE
if waveform.size(0) > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
waveform = waveform.squeeze(0)
waveform = waveform / (waveform.abs().max() + 1e-9)

times = np.arange(len(waveform)) / SAMPLE_RATE
fig, ax = plt.subplots(figsize=(16, 3))
ax.plot(times, waveform.numpy(), color='gray', linewidth=0.6)
colors = {index: item for index, item in enumerate(mcolors.TABLEAU_COLORS)}
for start, end, label in segments_ref:
    ax.axvspan(
        start,
        end,
        color=colors.get(label),
        alpha=0.2,
        label=f'speaker_{label}',
    )
handles, legend_labels = ax.get_legend_handles_labels()
by_label = dict(zip(legend_labels, handles))
ax.legend(by_label.values(), by_label.keys())
ax.set_title('Waveform with reference speaker diarization markup')
ax.set_xlabel('Time, s')
ax.grid(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
spk_model = ResNet(
    BasicBlock,
    layers=[3, 4, 6, 3],
    activation=torch.nn.ReLU,
    num_filters=[32, 64, 128, 256],
    nOut=1211,
    encoder_type='SP',
    n_mels=40,
    log_input=True,
)
checkpoint = torch.load(
    DATA_DIR / 'lab3_models' / 'lab3_model_0039.pth',
    map_location=device,
)
model_state = {
    name[6:]: value
    for name, value in checkpoint['model'].items()
    if name.startswith('__S__.')
}
spk_model.load_state_dict(model_state)
spk_model.to(device).eval()

win_len = int(WINDOW_SEC * SAMPLE_RATE)
step = int(STEP_SEC * SAMPLE_RATE)
frames = []
starts = []
for start in range(0, max(1, len(waveform) - win_len + 1), step):
    end = start + win_len
    if end > len(waveform):
        chunk = torch.zeros(win_len)
        chunk[:len(waveform) - start] = waveform[start:]
    else:
        chunk = waveform[start:end]
    frames.append(chunk.unsqueeze(0))
    starts.append(start)

frames = torch.stack(frames, dim=0)
print('Total frames:', frames.shape[0])

embeddings = []
with torch.no_grad():
    for index in range(0, len(frames), 64):
        batch = frames[index:index + 64].to(device).squeeze(1)
        embeddings.append(spk_model(batch).cpu())

embeddings = torch.cat(embeddings, dim=0).numpy()
print('Embedding tensor shape:', embeddings.shape)

labels = run_clustering(embeddings)
print('Clusters found:', sorted(set(labels.tolist())))
segments_hyp = merge_labels_to_segments(labels, starts, win_len, SAMPLE_RATE)
for start, end, label in segments_hyp:
    print(f'{start:.2f}-{end:.2f}s \t speaker_{label}')
print('Total number of segments:', len(segments_hyp))

fig, ax = plt.subplots(figsize=(16, 3))
ax.plot(times, waveform.numpy(), color='gray', linewidth=0.6)
for start, end, label in segments_hyp:
    ax.axvspan(
        start,
        end,
        color=colors.get(label, 'tab:green'),
        alpha=0.2,
        label=f'speaker_{label}',
    )
handles, legend_labels = ax.get_legend_handles_labels()
by_label = dict(zip(legend_labels, handles))
ax.legend(by_label.values(), by_label.keys())
ax.set_title('Waveform with hypothesized speaker diarization markup')
ax.set_xlabel('Time, s')
ax.grid(True)

hyp_rttm = SYNTH_OUTPUT.with_name(SYNTH_OUTPUT.stem + '_hyp.rttm')
hyp_lines = []
for start, end, label in segments_hyp:
    duration = max(0.0, end - start)
    hyp_lines.append(
        f'SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} '
        f'<NA> <NA> speaker_{label} <NA> <NA>'
    )
hyp_rttm.write_text('\n'.join(hyp_lines))

subprocess.run(
    [
        sys.executable,
        str(LAB_DIR / 'score.py'),
        '-r',
        str(ref_rttm),
        '-s',
        str(hyp_rttm),
    ],
    cwd=LAB_DIR,
    check=True,
)

plt.show()
