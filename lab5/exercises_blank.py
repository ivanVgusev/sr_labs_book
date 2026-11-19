import itertools
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels)
    return scores[labels == 1], scores[labels == 0]


def plot_histograms_2sets(
    all_scores_1,
    all_labels_1,
    all_scores_2,
    all_labels_2,
    names=("in-domain", "out-of-domain"),
):
    target_1, impostor_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    target_2, impostor_2 = get_tar_imp_scores(all_scores_2, all_labels_2)
    score_sets = (target_1, impostor_1, target_2, impostor_2)

    if any(scores.size == 0 for scores in score_sets):
        raise ValueError("target and impostor score sets must be non-empty")

    lower = min(scores.min() for scores in score_sets)
    upper = max(scores.max() for scores in score_sets)
    colors = ("green", "red", "blue", "cyan")
    labels = [
        f"{domain}_{kind}"
        for domain, kind in itertools.product(names, ("tar", "imp"))
    ]

    for scores, color, label in zip(score_sets, colors, labels):
        bins = max(10, int(sqrt(scores.size)))
        plt.hist(
            scores,
            bins,
            histtype="step",
            color=color,
            range=(lower, upper),
            label=label,
        )

    plt.xlabel("$s$")
    plt.ylabel("$\\widehat{W}(s|H_0)$, $\\widehat{W}(s|H_1)$")
    plt.title("VoxCeleb1-O (cleaned), histograms")
    plt.legend()
    plt.grid()
    plt.show()

    eer_1, threshold_1 = get_eer(target_1, impostor_1)
    eer_2, threshold_2 = get_eer(target_2, impostor_2)
    print(
        f"Equal Error Rate {names[0]} (EER): {eer_1:.3f}%, "
        f"threshold EER: {threshold_1:.3f}"
    )
    print(
        f"Equal Error Rate {names[1]} (EER): {eer_2:.3f}%, "
        f"threshold EER: {threshold_2:.3f}"
    )


def _vector(embedding):
    values = torch.as_tensor(embedding).detach().cpu().float()
    return values.reshape(-1, values.shape[-1]).mean(dim=0).numpy()


def _matrix(data, keys):
    return np.stack([_vector(data[key]) for key in keys])


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(matrix.dtype).eps)


def _cohort_stats(queries, cohort, top_size, query_batch_size=64):
    means = np.empty(len(queries), dtype=np.float64)
    standard_deviations = np.empty(len(queries), dtype=np.float64)

    for start in range(0, len(queries), query_batch_size):
        stop = start + query_batch_size
        scores = queries[start:stop] @ cohort.T
        top_scores = np.partition(scores, -top_size, axis=1)[:, -top_size:]
        means[start:stop] = top_scores.mean(axis=1)
        standard_deviations[start:stop] = top_scores.std(axis=1)

    return means, standard_deviations


def mean_embd_norm(test_embds, adapt_embds):
    if not adapt_embds:
        raise ValueError("adaptation embeddings must be non-empty")

    adaptation_mean = _matrix(adapt_embds, list(adapt_embds)).mean(axis=0)
    return {
        key: torch.as_tensor(_vector(embedding) - adaptation_mean).unsqueeze(0)
        for key, embedding in test_embds.items()
    }


def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    trials = [line.strip().split() for line in lines]

    if not trials:
        return [], [], []
    if not adapt_data:
        raise ValueError("adaptation embeddings must be non-empty")
    if N_s <= 0:
        raise ValueError("N_s must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    trial_keys = sorted({key for _, enroll, test in trials for key in (enroll, test)})
    cohort_keys = list(adapt_data)
    trial_matrix = _normalize(_matrix(test_data, trial_keys))
    cohort_matrix = _normalize(_matrix(adapt_data, cohort_keys))
    top_size = min(N_s, len(cohort_keys))
    means, standard_deviations = _cohort_stats(
        trial_matrix,
        cohort_matrix,
        top_size,
    )
    standard_deviations = np.maximum(standard_deviations, eps)
    positions = {key: index for index, key in enumerate(trial_keys)}

    normalized_scores = []
    labels = []
    trial_names = []

    for label, enroll, test in trials:
        enroll_index = positions[enroll]
        test_index = positions[test]
        raw_score = float(trial_matrix[enroll_index] @ trial_matrix[test_index])
        enroll_score = (
            raw_score - means[enroll_index]
        ) / standard_deviations[enroll_index]
        test_score = (
            raw_score - means[test_index]
        ) / standard_deviations[test_index]

        normalized_scores.append((enroll_score + test_score) / 2)
        labels.append(int(label))
        trial_names.append(f"{enroll} {test}")

    return normalized_scores, labels, trial_names


class CalibrationDataset(Dataset):
    def __init__(self, target_scores, impostor_scores):
        self.target_scores = torch.as_tensor(target_scores, dtype=torch.float32)
        self.impostor_scores = torch.as_tensor(impostor_scores, dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.target_scores, self.impostor_scores


class LinearCalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calib_params = nn.Linear(1, 1)
        nn.init.ones_(self.calib_params.weight)
        nn.init.zeros_(self.calib_params.bias)

    def forward(self, scores):
        return self.calib_params(scores.reshape(-1, 1).float())


class CalibrationLoss(nn.Module):
    def __init__(self, ptar=0.01):
        super().__init__()

        if not 0 < ptar < 1:
            raise ValueError("ptar must be between zero and one")

        self.ptar = ptar
        self.log_prior_odds = float(np.log(ptar / (1 - ptar)))

    def forward(self, target_llrs, nontarget_llrs):
        target_loss = F.softplus(
            -target_llrs - self.log_prior_odds
        ).mean()
        nontarget_loss = F.softplus(
            nontarget_llrs + self.log_prior_odds
        ).mean()
        return self.ptar * target_loss + (1 - self.ptar) * nontarget_loss


def train_calibration(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    verbose=False,
):
    device = next(model.parameters()).device
    history = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for target_scores, impostor_scores in train_loader:
            target_scores = target_scores.to(device)
            impostor_scores = impostor_scores.to(device)

            optimizer.zero_grad(set_to_none=True)
            target_llrs = model(target_scores)
            impostor_llrs = model(impostor_scores)
            loss = criterion(target_llrs, impostor_llrs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if scheduler is not None and scheduler[1] == "iteration":
                scheduler[0].step()

        epoch_loss /= len(train_loader)
        history.append(epoch_loss)

        if scheduler is not None and scheduler[1] == "epoch":
            scheduler[0].step()

        if verbose:
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:03d}: "
                f"loss={epoch_loss:.6f}, lr={learning_rate:.6f}"
            )

    return history
