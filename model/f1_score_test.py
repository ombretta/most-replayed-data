import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from sklearn import metrics
import math
from scipy.stats import rankdata

def compute_f1_score(pred: np.ndarray, target: np.ndarray):
    n_shots = 100
    top_percent = 15
    pred_100 = interpolate_pred(pred, n_shots)
    step = n_shots / pred.shape[0]
    x_space = np.linspace(0.5*step, n_shots-0.5*step, pred.shape[0])
    target_100 = interp1d(x_space, target, kind='nearest', assume_sorted=True, fill_value='extrapolate')(np.linspace(0.5, n_shots-0.5, n_shots))
    partition_elem = int(math.floor(top_percent * n_shots / 100))
    pred_100_bool = np.zeros_like(pred_100)
    target_100_bool = np.zeros_like(target_100)
    pred_100_bool[np.argpartition(pred_100, len(pred_100)-partition_elem-1)[-partition_elem:]] = 1
    target_100_bool[np.argpartition(target_100, len(target_100)-partition_elem-1)[-partition_elem:]] = 1

    f1_score = metrics.f1_score(target_100_bool, pred_100_bool)

    # TODO: IN CASE N_SHOTS != 100
    # x_integral = np.linspace(0, duration_ms, 10000)
    # shots_bins = np.arange(0, duration_ms, SHOT_DURATION_MS)
    # x_pdf = heat_markers_spline(x_integral)
    # bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x_integral, x_pdf, statistic='mean', bins=shots_bins)

    # shots_durations = bin_edges[1:]-bin_edges[:-1]
    # shots_durations = shots_durations.astype(int)
    # shots_selected = knapSack(math.floor(SUMMARY_DURATION_RATIO * duration_ms), shots_durations, bin_means, len(bin_means))
    
    return f1_score


def interpolate_pred(pred, n_shots = 100):
    if pred.shape[0] == n_shots:
        return pred
    step = n_shots / pred.shape[0]
    x_space = np.linspace(0.5*step, n_shots-0.5*step, pred.shape[0])
    pred_interp = interp1d(x_space, pred, kind='nearest', axis=0, assume_sorted=True, fill_value='extrapolate') # could use kind='linear' here
    x_integral = np.linspace(0,n_shots,num=5000)
    pred_100, pred_100_bin_edges, _ = binned_statistic(x_integral, pred_interp(x_integral), statistic='mean', bins=n_shots) # TODO check if bins range is end-exclusive
    return pred_100

def interpolate_features(features, n=100):
    step = features.shape[0] / n
    features_binned_mean = []
    for i in range(n):
        start_idx = max(int((i - 0.5) * step), 0)  # Adjust the start index to consider centered heat_markers
        end_idx = min(int((i + 0.5) * step), features.shape[0] -1)    # Adjust the end index to consider centered heat_markers
        bin_mean = np.mean(features[start_idx:end_idx], axis=0)
        features_binned_mean.append(bin_mean)
    features = np.array(features_binned_mean)
    return features


def build_gt_sorted_shots(x: np.ndarray):
    return np.argsort(x)

def sorted_shots_to_ranking(sorted_shots):
    return rankdata(sorted_shots).astype(int)

def top_k(gt_sorted_shots, sorted_shots, k=1):
    gt_top = gt_sorted_shots[-1]
    pred_top = set(sorted_shots[-k:])
    if gt_top in pred_top:
        return 1.0
    else:
        return 0.0

def precision_at_k(gt_sorted_shots, pred_sorted_shots, k=1):
    gt_ones = set(gt_sorted_shots[-k:])
    pred_ones = set(pred_sorted_shots[-k:])
    intersect = gt_ones.intersection(pred_ones)
    return float(len(intersect))/float(len(gt_ones))
