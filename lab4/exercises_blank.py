# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    tar_scores = all_scores[all_labels == 1]
    imp_scores = all_scores[all_labels == 0]
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    sort_indices = np.argsort(all_scores)
    all_scores_sort = np.asarray(all_scores)[sort_indices]
    ground_truth_sort = np.asarray(all_labels, dtype=bool)[sort_indices]

    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)

    eps = np.finfo(float).tiny
    LLR = np.log(np.maximum(tar_gauss_pdf, eps)) - np.log(np.maximum(imp_gauss_pdf, eps))
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    thresholds, fnr_thr, fpr_thr = _error_rates(ground_truth_sort, LLR)
    P_err = fnr_thr * P_Htar + fpr_thr * (1 - P_Htar)
    
    # Plot error's prob.
    plot(thresholds, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = P_err[P_err_idx]

    return thresholds[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    thresholds, fnr_values, fpr_values = _error_rates(ground_truth_sort, LLR)
    valid = np.flatnonzero(fnr_values <= fnr)

    if len(valid) == 0:
        idx = np.argmin(fnr_values)
    else:
        idx = valid[-1]

    return thresholds[idx], fpr_values[idx]

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    thr = _bayes_threshold(P_Htar, C00, C10, C01, C11)
    fnr, fpr = _rates_at_threshold(ground_truth_sort, LLR, thr)
    AC = _average_cost(fnr, fpr, P_Htar, C00, C10, C01, C11)

    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    costs = []

    for probability in P_Htar_thr:
        threshold = _bayes_threshold(probability, C00, C10, C01, C11)
        fnr, fpr = _rates_at_threshold(ground_truth_sort, LLR, threshold)
        costs.append(_average_cost(fnr, fpr, probability, C00, C10, C01, C11))

    costs = np.asarray(costs)
    candidates = np.flatnonzero(np.isclose(costs, np.max(costs)))
    idx = candidates[np.argmin(np.abs(np.asarray(P_Htar_thr)[candidates] - 0.5))]
    P_Htar = P_Htar_thr[idx]
    thr = _bayes_threshold(P_Htar, C00, C10, C01, C11)
    fnr, fpr = _rates_at_threshold(ground_truth_sort, LLR, thr)
    AC = costs[idx]

    return thr, fnr, fpr, AC, P_Htar


def _error_rates(ground_truth, LLR):
    ground_truth = np.asarray(ground_truth, dtype=bool)
    LLR = np.asarray(LLR)

    order = np.argsort(LLR)
    sorted_llr = LLR[order]
    sorted_labels = ground_truth[order]

    thresholds, last_indices = np.unique(sorted_llr, return_index=True)
    last_indices = np.r_[last_indices[1:] - 1, len(sorted_llr) - 1]

    thresholds = np.r_[np.nextafter(thresholds[0], -np.inf), thresholds]
    target_below = np.r_[0, np.cumsum(sorted_labels)[last_indices]]
    impostor_below = np.r_[0, np.cumsum(~sorted_labels)[last_indices]]

    target_count = np.sum(sorted_labels)
    impostor_count = np.sum(~sorted_labels)

    fnr = target_below / target_count
    fpr = (impostor_count - impostor_below) / impostor_count

    return thresholds, fnr, fpr


def _rates_at_threshold(ground_truth, LLR, threshold):
    ground_truth = np.asarray(ground_truth, dtype=bool)
    decisions = np.asarray(LLR) > threshold

    fnr = np.mean(~decisions[ground_truth])
    fpr = np.mean(decisions[~ground_truth])

    return fnr, fpr


def _bayes_threshold(P_Htar, C00, C10, C01, C11):
    numerator = (C01 - C11) * (1 - P_Htar)
    denominator = (C10 - C00) * P_Htar

    if numerator == 0:
        return -np.inf
    if denominator == 0:
        return np.inf
    if numerator < 0 or denominator < 0:
        raise ValueError('Costs do not define a valid Bayes threshold')

    return np.log(numerator / denominator)


def _average_cost(fnr, fpr, P_Htar, C00, C10, C01, C11):
    target_cost = C00 * (1 - fnr) + C10 * fnr
    impostor_cost = C01 * fpr + C11 * (1 - fpr)

    return target_cost * P_Htar + impostor_cost * (1 - P_Htar)
