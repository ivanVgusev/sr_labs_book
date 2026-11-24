#!/usr/bin/env python

import os
import sys
from collections import defaultdict
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
ROOT_DIR = LAB_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
sys.path.insert(0, str(LAB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import common
from common import (
    compute_scores_cosine,
    extract_features,
    get_dcf,
    get_eer,
    get_voxceleb_filelist,
    run_voxceleb_convert,
    test_dataset_loader,
)
from common.voxceleb_paths import (
    VOX1_DEV_WAV,
    VOX1_DEV_WAV_MP3,
    VOX1_TEST_WAV,
    VOX1_TEST_WAV_MP3,
    VOX1_VERI_TEST2,
    require_voxceleb,
)
from exercises_blank import (
    CalibrationDataset,
    CalibrationLoss,
    LinearCalibrationModel,
    get_tar_imp_scores,
    plot_histograms_2sets,
    s_norm,
    train_calibration,
)
from LoadModel import load_model
from ResNetSE34V2 import MainModel


require_voxceleb(VOX1_DEV_WAV, VOX1_TEST_WAV, VOX1_VERI_TEST2)
print(f'VoxCeleb1 dev: {VOX1_DEV_WAV}')
print(f'VoxCeleb1 test: {VOX1_TEST_WAV}')
print(f'VoxCeleb1 protocol: {VOX1_VERI_TEST2}')


def apply_mp3_codec(source_path, target_path):
    common.apply_mp3_codec(source_path, target_path)


run_voxceleb_convert(
    input_path=VOX1_TEST_WAV,
    result_path=VOX1_TEST_WAV_MP3,
    fun=apply_mp3_codec,
    threads=10,
)
run_voxceleb_convert(
    input_path=VOX1_DEV_WAV,
    result_path=VOX1_DEV_WAV_MP3,
    fun=apply_mp3_codec,
    threads=10,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MainModel()

with open(DATA_DIR / 'lists' / 'models.txt', 'r') as file:
    model_urls = file.readlines()

model = load_model(
    model,
    model_urls,
    save_path=DATA_DIR / 'models',
    device=device,
).to(device).eval()
print(f'Device: {device}')

with open(VOX1_VERI_TEST2, 'r') as file:
    protocol = file.readlines()

test_files = sorted({
    path
    for line in protocol
    for path in line.strip().split()[-2:]
})
num_workers = 0 if sys.platform == 'darwin' else 8

test_dataset = test_dataset_loader(
    test_files,
    test_path=VOX1_TEST_WAV,
    eval_frames=500,
    num_eval=1,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
)
features_out_of_domain = extract_features(model, test_loader)

test_dataset = test_dataset_loader(
    test_files,
    test_path=VOX1_TEST_WAV_MP3,
    eval_frames=500,
    num_eval=1,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
)
features_in_domain = extract_features(model, test_loader)

scores_in_domain, labels_in_domain, trials_in_domain = compute_scores_cosine(
    features_in_domain,
    protocol,
)
scores_out_of_domain, labels_out_of_domain, trials_out_of_domain = (
    compute_scores_cosine(features_out_of_domain, protocol)
)
plot_histograms_2sets(
    scores_in_domain,
    labels_in_domain,
    scores_out_of_domain,
    labels_out_of_domain,
)

adaptation_files = sorted(get_voxceleb_filelist(VOX1_DEV_WAV_MP3))
adaptation_dataset = test_dataset_loader(
    adaptation_files,
    test_path=VOX1_DEV_WAV_MP3,
    eval_frames=1000,
    num_eval=1,
)
adaptation_loader = DataLoader(
    adaptation_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
)
adaptation_features = extract_features(model, adaptation_loader)

normalized_scores, normalized_labels, normalized_trials = s_norm(
    features_in_domain,
    protocol,
    adaptation_features,
    N_s=100,
    eps=0.6,
)
plot_histograms_2sets(
    normalized_scores,
    normalized_labels,
    scores_out_of_domain,
    labels_out_of_domain,
)

target_prior = 0.5
calibration_protocol_size = 40_000
calibration_epochs = 300
calibration_learning_rate = 0.05
random_seed = 42

calibration_files = sorted(get_voxceleb_filelist(VOX1_DEV_WAV))
calibration_dataset = test_dataset_loader(
    calibration_files,
    test_path=VOX1_DEV_WAV,
    eval_frames=1000,
    num_eval=1,
)
calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
)
calibration_features = extract_features(model, calibration_loader)

rng = np.random.default_rng(random_seed)
utterances = defaultdict(lambda: defaultdict(list))

for path in calibration_features:
    speaker, session, _ = path.split('/')
    utterances[speaker][session].append(path)

speakers = sorted(utterances)
target_speakers = [
    speaker
    for speaker in speakers
    if len(utterances[speaker]) >= 2
]
target_count = int(target_prior * calibration_protocol_size)
impostor_count = calibration_protocol_size - target_count
calibration_protocol = []

for _ in range(target_count):
    speaker = rng.choice(target_speakers)
    sessions = rng.choice(list(utterances[speaker]), size=2, replace=False)
    enroll = rng.choice(utterances[speaker][sessions[0]])
    test = rng.choice(utterances[speaker][sessions[1]])
    calibration_protocol.append(f'1 {enroll} {test}')

for _ in range(impostor_count):
    enroll_speaker, test_speaker = rng.choice(speakers, size=2, replace=False)
    enroll_session = rng.choice(list(utterances[enroll_speaker]))
    test_session = rng.choice(list(utterances[test_speaker]))
    enroll = rng.choice(utterances[enroll_speaker][enroll_session])
    test = rng.choice(utterances[test_speaker][test_session])
    calibration_protocol.append(f'0 {enroll} {test}')

rng.shuffle(calibration_protocol)

calibration_scores, calibration_labels, calibration_trials = (
    compute_scores_cosine(calibration_features, calibration_protocol)
)
target_calibration_scores, impostor_calibration_scores = get_tar_imp_scores(
    calibration_scores,
    calibration_labels,
)
calibration_train_set = CalibrationDataset(
    target_calibration_scores,
    impostor_calibration_scores,
)
calibration_train_loader = DataLoader(calibration_train_set, batch_size=1)

calibration_model = LinearCalibrationModel()
calibration_loss = CalibrationLoss(ptar=target_prior)
optimizer = torch.optim.Adam(
    calibration_model.parameters(),
    lr=calibration_learning_rate,
)
scheduler = (
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1),
    'epoch',
)
loss_history = train_calibration(
    calibration_train_loader,
    calibration_model,
    calibration_loss,
    optimizer,
    scheduler,
    calibration_epochs,
)

weight = calibration_model.calib_params.weight.item()
bias = calibration_model.calib_params.bias.item()
print(f'Final loss: {loss_history[-1]:.6f}')
print(f'Calibration: score -> {weight:.4f} * score + {bias:.4f}')

checkpoint_dir = DATA_DIR / 'lab5_models'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / 'lab5_calibration.pth'
torch.save(
    {
        'model': calibration_model.state_dict(),
        'target_prior': target_prior,
        'epochs': calibration_epochs,
    },
    checkpoint_path,
)
print(f'Calibration model saved to: {checkpoint_path}')

target_test_scores, impostor_test_scores = get_tar_imp_scores(
    scores_out_of_domain,
    labels_out_of_domain,
)

calibration_model.eval()
with torch.no_grad():
    calibrated_target_scores = calibration_model(
        torch.as_tensor(target_test_scores, dtype=torch.float32)
    ).squeeze().numpy()
    calibrated_impostor_scores = calibration_model(
        torch.as_tensor(impostor_test_scores, dtype=torch.float32)
    ).squeeze().numpy()

lower = min(calibrated_target_scores.min(), calibrated_impostor_scores.min())
upper = max(calibrated_target_scores.max(), calibrated_impostor_scores.max())
plt.hist(
    calibrated_target_scores,
    max(10, int(np.sqrt(len(calibrated_target_scores)))),
    histtype='step',
    color='blue',
    range=(lower, upper),
    label='target',
)
plt.hist(
    calibrated_impostor_scores,
    max(10, int(np.sqrt(len(calibrated_impostor_scores)))),
    histtype='step',
    color='cyan',
    range=(lower, upper),
    label='impostor',
)
plt.xlabel('$s_c$')
plt.ylabel('$\\widehat{W}(s_c|H_0)$, $\\widehat{W}(s_c|H_1)$')
plt.title('VoxCeleb1-O (cleaned), calibrated scores')
plt.legend()
plt.grid()
plt.show()

eer_before, _ = get_eer(target_test_scores, impostor_test_scores)
min_dcf_before, _, act_dcf_before, _ = get_dcf(
    target_test_scores,
    impostor_test_scores,
    P_target=target_prior,
)
print(
    f'Before calibration: EER={eer_before:.3f}%, '
    f'minDCF={min_dcf_before:.4f}, actDCF={act_dcf_before:.4f}'
)

eer_after, _ = get_eer(
    calibrated_target_scores,
    calibrated_impostor_scores,
)
min_dcf_after, _, act_dcf_after, _ = get_dcf(
    calibrated_target_scores,
    calibrated_impostor_scores,
    P_target=target_prior,
)
print(
    f'After calibration: EER={eer_after:.3f}%, '
    f'minDCF={min_dcf_after:.4f}, actDCF={act_dcf_after:.4f}'
)
