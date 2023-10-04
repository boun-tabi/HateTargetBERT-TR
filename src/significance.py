import pandas as pd
from pathlib import Path



from sklearn.metrics import f1_score
import numpy as np
def bootstrap_test_predictions(all_preds_model_A, all_preds_model_B, all_ground_truth, num_iterations=1000, alpha=0.05):
    bootstrap_diffs = []
    for _ in range(num_iterations):
        # Resample indices with replacement
        indices = np.random.choice(len(all_ground_truth), size=len(all_ground_truth), replace=True)
        resampled_truth = all_ground_truth[indices]
        resampled_pred_A = all_preds_model_A[indices]
        resampled_pred_B = all_preds_model_B[indices]

        f1_A = f1_score(resampled_truth, resampled_pred_A)
        f1_B = f1_score(resampled_truth, resampled_pred_B)

        diff = f1_A - f1_B
        bootstrap_diffs.append(diff)

    # Calculate the confidence interval
    lower_bound = np.percentile(bootstrap_diffs, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


def check_significance(all_preds_model_A, all_preds_model_B, all_ground_truth, alpha=0.05):
  lower_bound, upper_bound = bootstrap_test_predictions(all_preds_model_A, all_preds_model_B, all_ground_truth)
  print(f"{100*(1-alpha)}% Confidence Interval for difference: ({lower_bound:.4f}, {upper_bound:.4f})")

  if lower_bound > 0 or upper_bound < 0:
      print("The difference is statistically significant.")
  else:
      print("The difference is not statistically significant.")

base_path = '../checkpoints'
all_labels = {'BERTurk': [], 'HateTargetBERT': [], 'GT': []}
for i in range(10):
  hatetargetbert_metrics = pd.read_json(f'{base_path}/HateTargetBERT_run{i}_TA_TS_PRE_POST_MN/test_metrics.json')
  berturk_metrics = pd.read_json(f'{base_path}/BERTurk_run{i}/test_metrics.json')
  assert all([i == j for i,j in zip(hatetargetbert_metrics['labels'].tolist(), berturk_metrics['labels'].tolist())])
  all_labels['BERTurk'].extend(berturk_metrics['predictions'])
  all_labels['HateTargetBERT'].extend(hatetargetbert_metrics['predictions'])
  all_labels['GT'].extend(berturk_metrics['labels'])

# Example: concatenate predictions and ground truth from 5 splits
all_ground_truth = np.array(all_labels['GT'])
all_preds_model_A = np.array(all_labels['HateTargetBERT'])
all_preds_model_B = np.array(all_labels['BERTurk'])

print('HateTargetBERT vs. BERTurk')
print('--------------------------')
print('Overall')
check_significance(all_preds_model_A, all_preds_model_B, all_ground_truth)


for i in range(10):
   print(f'Split {i}')
   pred_A = all_preds_model_A[i*669:(i+1)*669]
   pred_B = all_preds_model_B[i*669:(i+1)*669]
   GT = all_ground_truth[i*669:(i+1)*669]
   check_significance(pred_A, pred_B, GT)

