
from model.f1_score_test import compute_f1_score
from sklearn.metrics import mean_squared_error
import random
import numpy as np
from scipy.interpolate import interp1d
import h5py
import json

criterion = mean_squared_error

def test_random_scores(num_tests = 10000, interpolate_frame_features = False, mode='uniform', gaussian_params = (0.5, 0.5)):

    filename = './data/datasets/yt/out.h5'
    splits_filename = ['./data/datasets/splits/yt_splits.json']
    split_index = 0

    hdf = h5py.File(filename, 'r')
    # self.list_frame_features, self.list_gtscores = [], []

    f1_scores = []
    losses_test = []

    with open(splits_filename[0]) as f:
        data = json.loads(f.read())
        for i, sp in enumerate(data):
            if i == split_index:
                split = sp
                break

    dataset_mode = 'train'
    for video_name in split[dataset_mode + '_keys']:
        frame_features = np.array(hdf[video_name + '/features'])
        heat_markers = np.array(hdf[video_name + '/heat-markers'])
        interpolation_num = heat_markers.shape[0] # maybe + 1

        if mode == 'zeros': 
            generation_fn = lambda n: np.zeros(n)
        elif mode == 'uniform':
            generation_fn = lambda n: np.random.uniform(0,1, n)
        elif mode == 'gaussian' or mode == 'gaussian_noise':
            generation_fn = lambda n: np.random.normal(gaussian_params[0], gaussian_params[1], n)
        elif mode == 'original':
            generation_fn = lambda n: np.zeros(n)
        else:
            print("Mode error")
            exit(1)

        
        for i in range(num_tests):

            interpolation_num = heat_markers.shape[0]
            heat_markers_space = np.linspace(0.5, heat_markers.shape[0]-0.5, heat_markers.shape[0])
            step = heat_markers.shape[0] / frame_features.shape[0]
            frame_features_heat_markers_space = np.linspace(0.5*step, heat_markers.shape[0]-0.5*step, frame_features.shape[0])
            if interpolate_frame_features:
                scores_np = generation_fn(heat_markers.shape[0])
                gtscore_np = heat_markers
                # xp = frame_features_heat_markers_space
                # x = heat_markers_space
                # frame_features = interp1d(xp, frame_features, axis=0, assume_sorted=True, fill_value='extrapolate')(x)
            else:
                scores_np = generation_fn(frame_features.shape[0])
                xp = heat_markers_space
                x = frame_features_heat_markers_space
                gtscore_np = interp1d(xp, heat_markers, kind='nearest', assume_sorted=True, fill_value='extrapolate')(x)

            if mode == 'gaussian_noise' or mode == 'original':
                scores_np += gtscore_np

            loss_test = criterion(scores_np, gtscore_np)
            f1_score = compute_f1_score(scores_np, gtscore_np)
            f1_scores.append(f1_score)
            losses_test.append(loss_test.data)
        
    return np.stack(f1_scores).mean(), np.stack(losses_test).mean()

print("-----ORIGINAL-----")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1, interpolate_frame_features=False, mode='original')
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1, interpolate_frame_features=True, mode='original')
print(f"f1_score: {f1_score}, mse: {mse}")
print("----UNIFORM----")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False)
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True)
print(f"f1_score: {f1_score}, mse: {mse}")
print("-----ZEROS-----")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False, mode='zeros')
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True, mode='zeros')
print(f"f1_score: {f1_score}, mse: {mse}")
print("---GAUSSIAN std=0.5---")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False, mode='gaussian', gaussian_params=(0.5, 0.5))
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True, mode='gaussian', gaussian_params=(0.5, 0.5))
print(f"f1_score: {f1_score}, mse: {mse}")
print("---GAUSSIAN std=0.2---")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False, mode='gaussian', gaussian_params=(0.5, 0.2))
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True, mode='gaussian', gaussian_params=(0.5, 0.2))
print(f"f1_score: {f1_score}, mse: {mse}")
print("--GAUSSIAN NOISE std=0.1--")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False, mode='gaussian_noise', gaussian_params=(0.0, 0.1))
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True, mode='gaussian_noise', gaussian_params=(0.0, 0.1))
print(f"f1_score: {f1_score}, mse: {mse}")
print("--GAUSSIAN NOISE std=0.2--")
print("With interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=False, mode='gaussian_noise', gaussian_params=(0.0, 0.2))
print(f"f1_score: {f1_score}, mse: {mse}")
print("Without interpolation before training:")
f1_score, mse = test_random_scores(1000, interpolate_frame_features=True, mode='gaussian_noise', gaussian_params=(0.0, 0.2))
print(f"f1_score: {f1_score}, mse: {mse}")

