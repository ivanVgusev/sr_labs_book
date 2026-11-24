# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram

    tar_scores = []
    imp_scores = []

    for idx in range(len(all_labels)):
        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])
        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio

    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)

    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')

    sort_idx = np.argsort(all_scores)
    all_scores_sort = np.array(all_scores)[sort_idx]
    ground_truth_sort = np.array(all_labels, dtype='bool')[sort_idx]

    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))

    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf + 1e-10) - np.log(imp_gauss_pdf + 1e-10)

    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test

    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision

        err = (solution != ground_truth_sort)                          # error vector

        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)

        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error

    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()

    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)

    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test

    thr   = 0.0
    fpr   = 1.0

    for idx in range(len(LLR)):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)

        fnr_thr = np.sum(err[ground_truth_sort])/len(tar_scores)
        fpr_thr = np.sum(err[~ground_truth_sort])/len(imp_scores)

        if fnr_thr <= fnr and fpr_thr < fpr:
            thr = LLR[idx]
            fpr = fpr_thr

    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test

    if P_Htar == 0:
        thr = np.inf
    elif P_Htar == 1:
        thr = -np.inf
    else:
        thr = np.log((C01 - C11)*(1 - P_Htar)/((C10 - C00)*P_Htar))

    solution = LLR > thr
    err = (solution != ground_truth_sort)

    fnr = np.sum(err[ground_truth_sort])/len(tar_scores)
    fpr = np.sum(err[~ground_truth_sort])/len(imp_scores)

    AC = P_Htar*(C00*(1 - fnr) + C10*fnr) + \
         (1 - P_Htar)*(C01*fpr + C11*(1 - fpr))

    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test

    AC_thr = np.zeros(len(P_Htar_thr))

    for idx in range(len(P_Htar_thr)):
        _, _, _, AC_thr[idx] = bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores,
                                          P_Htar_thr[idx], C00, C10, C01, C11)

    AC_idx = np.argmax(AC_thr)
    P_Htar = P_Htar_thr[AC_idx]
    thr, fnr, fpr, AC = bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores,
                                   P_Htar, C00, C10, C01, C11)

    return thr, fnr, fpr, AC, P_Htar
