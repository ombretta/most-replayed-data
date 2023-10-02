import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np
import math
from evaluation.user_study.ranking_from_comparisons import build_sorted_shots
from model.f1_score_test import interpolate_pred, build_gt_sorted_shots, sorted_shots_to_ranking, precision_at_k, top_k
from scipy.stats import kendalltau
import krippendorff


parser = argparse.ArgumentParser()
parser.add_argument("csv_inputs", type=Path)
parser.add_argument('--json_path', help="path to a .json file with results from the model")
parser.add_argument('--video_heatmarkers_dir', required=True, help="path to the directory with heat-markers video-id.h5 files")
parser.add_argument('--validate', action='store_true')
args = parser.parse_args()


def load_pred(video_id, path, n_shots=10):
    with open(path, 'r') as f:
        pred = json.load(f)
        current_pred = pred[video_id]
        pred_10 = interpolate_pred(np.array(current_pred), n_shots)
        return pred_10

def load_gt(video_id, n_shots=10):
    try:
        heat_markers_pd = pd.read_hdf(args.video_heatmarkers_dir+'/'+video_id+'.h5')
    except:
        print(f"Error: {video_id}, .h5 not found")
    gt_10 = interpolate_pred(np.array(heat_markers_pd.heatMarkerIntensityScoreNormalized), n_shots)
    return gt_10

class TaskAnswers:
    N_COMPARISONS = 20
    N_SHOTS = 10
    def __init__(self, s: str):
        answers = json.loads(s)
        raw_dict = answers[0]
        self.comparisons = self._parse_mturk_format(raw_dict)

    def _parse_mturk_format(self, raw_dict):
        comparisons = []
        for i in range(self.N_COMPARISONS):
            left = raw_dict[f"pair{i}_0"]
            right = raw_dict[f"pair{i}_1"]
            num_choices = 0
            if raw_dict[f"{i}_left"]["Left"] == True: 
                num_choices += 1
                user_choice = 1 # left
            if raw_dict[f"{i}_right"]["Right"] == True:
                num_choices += 1
                user_choice = -1 # right
            if raw_dict[f"{i}_control"]["CONTROL"] == True:
                num_choices += 1
                user_choice = "CONTROL"
            if num_choices !=1:
                user_choice = "BROKEN"
            comparisons.append({"left": left, "right": right, "user_choice": user_choice})
        return comparisons

    def validate(self) -> bool:
        if not self.comparisons:
            raise Exception("Cannot validate, missing data")
            return
        for i,pair in enumerate(self.comparisons):
            if pair["user_choice"] == "BROKEN":
                # print("broken radio")
                return False
            if (pair["left"] == "CONTROL" or pair["right"] == "CONTROL") != (pair["user_choice"] == "CONTROL"):
                # print("missed CONTROL")
                return False
        return True

    # only call on validated data
    def get_clean_comparisons(self):
        return list(map(lambda pair: (pair["left"], pair["right"], pair["user_choice"]), filter(lambda pair: pair["user_choice"] != "CONTROL", self.comparisons)))

    def full_comparison_row(self):
        n_entries = self.N_SHOTS * (self.N_SHOTS - 1) / 2 # 45 = [0,9] x [1,9]
        result = a = np.full([self.N_SHOTS, self.N_SHOTS], np.nan)
        for comparison in self.get_clean_comparisons():
            left = int(comparison[0])
            right = int(comparison[1])
            user_choice = int(comparison[2])
            if left > right:
                i = right
                j = left
                value = -user_choice
            elif right > left:
                i = left
                j = right
                value = user_choice
            result[i,j] = value
        triu = np.triu_indices(n=self.N_SHOTS,k=1)
        result = result[triu]
        assert(len(result) == n_entries)
        return result

    def __str__(self):
        return "\n".join(map(lambda x: x.__str__(), self.comparisons))
    
#for each video id
def group_fn(df_group):
    answs = df_group['Answer.taskAnswers']
    video_id = str(df_group.head(1)['Input.video_id'].iat[0])
    print("Loading gt and pred for ", video_id)
    gt_10 = load_gt(video_id)
    print(gt_10)
    gt_sorted_shots = build_gt_sorted_shots(gt_10)
    print(gt_sorted_shots)
    gt_ranking = sorted_shots_to_ranking(gt_sorted_shots)
    print("gt ranking:  ", gt_ranking)

    pred_taus = []
    pred_top1s = []
    pred_top3s = []
    pred_top5s = []
    pred_precision3s = []
    pred_precision5s = []
  
    for epoch in range(250, 300):
        pred_10 = load_pred(video_id, path=args.json_path+"/"+"yt_500-test_"+str(epoch)+".json")
        pred_sorted_shots = build_gt_sorted_shots(pred_10)
        # print(pred_sorted_shots)
        pred_ranking = sorted_shots_to_ranking(pred_sorted_shots)
        # print("pred ranking:",pred_ranking)
        pred_top1 = top_k(gt_sorted_shots, pred_sorted_shots, k=1)
        pred_top3 = top_k(gt_sorted_shots, pred_sorted_shots, k=3)
        pred_top5 = top_k(gt_sorted_shots, pred_sorted_shots, k=5)
        pred_precision3 = precision_at_k(gt_sorted_shots, pred_sorted_shots, k=3)
        pred_precision5 = precision_at_k(gt_sorted_shots, pred_sorted_shots, k=5)
        pred_tau = kendalltau(gt_ranking, pred_ranking)
        
        pred_top1s.append(pred_top1)
        pred_top3s.append(pred_top3)
        pred_top5s.append(pred_top5)
        pred_precision3s.append(pred_precision3)
        pred_precision5s.append(pred_precision5)
        pred_taus.append(pred_tau)
        # print(pred_tau)
        

    
    taus = []
    top1s = []
    top3s = []
    top5s = []
    precision3s = []
    precision5s = []
    full_comparison_rows = []
    rankings = []
    for ans in answs:
        ans = TaskAnswers(ans)
        # print(ans)
        sorted_shots = np.array([int(x) for x in build_sorted_shots(ans.get_clean_comparisons())])
        full_comparison_rows.append(ans.full_comparison_row())
        # print(sorted_shots)
        ranking = sorted_shots_to_ranking(sorted_shots)
        rankings.append(ranking)
        # print(ranking)
        tau = kendalltau(gt_ranking, ranking)
        # print(tau)
        taus.append(tau.statistic)
        # print(sorted_shots[-1])
        top1 = top_k(gt_sorted_shots, sorted_shots, k=1)
        top1s.append(top1)
        top3 = top_k(gt_sorted_shots, sorted_shots, k=3)
        top3s.append(top3)
        top5 = top_k(gt_sorted_shots, sorted_shots, k=5)
        top5s.append(top5)
        precision3 = precision_at_k(gt_sorted_shots, sorted_shots, k=3)
        precision3s.append(precision3)
        precision5 = precision_at_k(gt_sorted_shots, sorted_shots, k=5)
        precision5s.append(precision5)

    comparisons = np.array(full_comparison_rows)
    alpha_comparisons = krippendorff.alpha(reliability_data=comparisons, level_of_measurement="nominal")
    rankings = np.array(rankings)
    alpha_rankings = krippendorff.alpha(reliability_data=rankings, level_of_measurement="ordinal")
    result = pd.DataFrame([
        (video_id, int(answs.shape[0]), alpha_comparisons, alpha_rankings, 
            np.array(taus).mean(), np.array(taus).std(), 
            np.array(top1s).mean(), np.array(top1s).std(), 
            np.array(top3s).mean(),  np.array(top3s).std(),
            np.array(top5s).mean(), np.array(top5s).std(),
            np.array(precision3s).mean(), np.array(precision3s).std(),
            np.array(precision5s).mean(), np.array(precision5s).std(),

            np.array(pred_taus).mean(), np.array(pred_taus).std(),
            np.array(pred_top1s).mean(), np.array(pred_top1s).std(),
            np.array(pred_top3s).mean(), np.array(pred_top3s).std(),
            np.array(pred_top5s).mean(), np.array(pred_top5s).std(),
            np.array(pred_precision3s).mean(), np.array(pred_precision3s).std(),
            np.array(pred_precision5s).mean(), np.array(pred_precision5s).std()
        )], 
        columns=[
            'Input.video_id', 'n_workers', 'krippendorff', 'krippendorff_rank', 
            'tau_avg', 'tau_std', 
            'top-1_avg', 'top-1_std', 
            'top-3_avg', 'top-3_std', 
            'top-5_avg', 'top-5_std',
            'precision@3_avg', 'precision@3_std', 
            'precision@5_avg', 'precision@5_std', 
            'pred_tau', 'pred_tau_std', 
            'pred_top-1', 'pred_top-1_std',
            'pred_top-3', 'pred_top-3_std',
            'pred_top-5', 'pred_top-5_std',
            'pred_precision@3',  'pred_precision@3_std',
            'pred_precision@5', 'pred_precision@5_std'
        ])
    return result

if __name__ == '__main__':
    # for file in csv_inputs
    file = args.csv_inputs
    df = pd.read_csv(file)

    if args.validate:
        answs = [(TaskAnswers(row['Answer.taskAnswers']), row['WorkerId'], row['Input.video_id'], row['RejectionTime']) for k, row in df.iterrows()]
        print("Number of answers: ", len(answs))
        def validate_explicit(x):
            valid = x[0].validate()
            if not valid:
                if pd.isnull(x[3]):
                    print(x[1])
            return valid
        valid_answs = list(filter(lambda x: validate_explicit(x), answs))
        print("Number of valid answers: ", len(valid_answs))
    else:
        # 1. validation
        # TODO check user ids for rejection
        valid_idx = df.apply(lambda x: TaskAnswers(x['Answer.taskAnswers']).validate() and pd.isnull(x['RejectionTime']), axis=1)
        valid_df = df.loc[valid_idx]
        # 2. process data
        grouped_df = valid_df.groupby("Input.video_id", as_index=False).apply(group_fn).reset_index(drop=True)

        weighted_avg_df = pd.DataFrame()
        weighted_avg_df["Input.video_id"] = ["AVG"]
        weighted_avg_df['n_workers'] = grouped_df['n_workers'].sum()

        # Compute weighted average for each column
        for column in ['krippendorff', 'krippendorff_rank', 'tau_avg', 'top-1_avg', 'top-3_avg', 'top-5_avg', 'precision@3_avg', 'precision@5_avg']:
            weighted_avg = grouped_df[column].mul(grouped_df['n_workers']).sum() / grouped_df['n_workers'].sum()
            weighted_avg_df[column] = [weighted_avg]

        # TODO compute average without workers weight
        # 'pred_tau_avg', 'pred_top-1', 'pred_top-3', 'pred_top-5', 'pred_precision@3', 'pred_precision@5'

        for column in filter(lambda x: '_std' in x, grouped_df.columns):
            weighted_std = grouped_df[column].pow(2).mul(grouped_df['n_workers'] - 1).sum() / (grouped_df['n_workers'].sum() + grouped_df.shape[0])
            weighted_std = math.sqrt(weighted_std)
            weighted_avg_df[column] = [weighted_std]

        for column in ['pred_tau', 'pred_top-1', 'pred_top-3', 'pred_top-5', 'pred_precision@3', 'pred_precision@5']:
            weighted_avg_df[column] = grouped_df[column].mean()
            weighted_avg_df[column+"_std"] = grouped_df[column].std()


        # Append the weighted average row to the original dataframe
        grouped_avg_df = pd.concat([grouped_df, weighted_avg_df])
        print(grouped_avg_df)
        grouped_avg_df.to_csv('./evaluation/user_study/results/grouped_avg_df_102.csv')
