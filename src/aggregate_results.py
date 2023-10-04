import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats


def load_data(base_path, num_splits=10):
    all_labels = {'BERTurk': [], 'HateTargetBERT': [], 'GT': [], 'HateTargetNN': []}
    
    for i in range(num_splits):
        hatetargetbert_metrics = pd.read_json(f'{base_path}/HateTargetBERT_run{i}_TA_TS_PRE_POST_MN/test_metrics.json')
        berturk_metrics = pd.read_json(f'{base_path}/BERTurk_run{i}/test_metrics.json')
        hatetargetnn_metrics = pd.read_json(f'{base_path}/HateTargetNN_run{i}_TA_TS_PRE_POST_MN/test_metrics.json')
        
        assert all([i == j for i,j in zip(hatetargetbert_metrics['labels'].tolist(), berturk_metrics['labels'].tolist())])
        
        all_labels['BERTurk'].extend(berturk_metrics['predictions'])
        all_labels['HateTargetBERT'].extend(hatetargetbert_metrics['predictions'])
        all_labels['GT'].extend(berturk_metrics['labels'])
        all_labels['HateTargetNN'].extend(hatetargetnn_metrics['predictions'])
    
    return np.array(all_labels['GT']), np.array(all_labels['HateTargetBERT']), np.array(all_labels['BERTurk']), np.array(all_labels['HateTargetNN'])


def compute_metrics(true, pred):
    return accuracy_score(true, pred), precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)


def bootstrap_test_predictions(all_preds_model_A, all_preds_model_B, all_ground_truth, num_iterations=1000, alpha=0.05):
    bootstrap_diffs = []
    for _ in range(num_iterations):
        indices = np.random.choice(len(all_ground_truth), size=len(all_ground_truth), replace=True)
        resampled_truth = all_ground_truth[indices]
        resampled_pred_A = all_preds_model_A[indices]
        resampled_pred_B = all_preds_model_B[indices]

        diff = f1_score(resampled_truth, resampled_pred_A) - f1_score(resampled_truth, resampled_pred_B)
        bootstrap_diffs.append(diff)

    lower_bound = np.percentile(bootstrap_diffs, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


def check_significance(all_preds_model_A, all_preds_model_B, all_ground_truth, alpha=0.05):
    lower_bound, upper_bound = bootstrap_test_predictions(all_preds_model_A, all_preds_model_B, all_ground_truth)
    print(f"{100*(1-alpha)}% Confidence Interval for difference: ({lower_bound:.4f}, {upper_bound:.4f})")
    
    if lower_bound > 0 or upper_bound < 0:
        return True
    return False


# Load data
all_ground_truth, all_preds_model_A, all_preds_model_B, all_preds_model_C = load_data('../checkpoints')


# Check significance for overall data and each split
print('HateTargetBERT vs. BERTurk\n--------------------------')
print('Overall')
is_significant = check_significance(all_preds_model_A, all_preds_model_B, all_ground_truth)
print("Statistically significant difference." if is_significant else "No significant difference.")

split_size = len(all_ground_truth) // 10

scores = []
for i in range(10):
    print(f'\nSplit {i}')
    start, end = i * split_size, (i + 1) * split_size
    is_significant = check_significance(all_preds_model_A[start:end], all_preds_model_B[start:end], all_ground_truth[start:end])
    print("Statistically significant difference." if is_significant else "No significant difference.")
    # Compute overall metrics for both models
    acc_A, precision_A, recall_A, f1_A = compute_metrics(all_ground_truth[start:end], all_preds_model_A[start:end])
    acc_B, precision_B, recall_B, f1_B = compute_metrics(all_ground_truth[start:end], all_preds_model_B[start:end])
    acc_C, precision_C, recall_C, f1_C = compute_metrics(all_ground_truth[start:end], all_preds_model_C[start:end])
    scores.append({'model': 'HateTargetBERT', 'split': i, 'accuracy': acc_A, 'precision': precision_A, 'recall': recall_A, 'f1': f1_A})
    scores.append({'model': 'BERTurk', 'split': i, 'accuracy': acc_B, 'precision': precision_B, 'recall': recall_B, 'f1': f1_B})
    scores.append({'model': 'HateTargetNN', 'split': i, 'accuracy': acc_C, 'precision': precision_C, 'recall': recall_C, 'f1': f1_C})
    print(f"Model A - Accuracy: {acc_A:.4f}, Precision: {precision_A:.4f}, Recall: {recall_A:.4f},F1 Score: {f1_A:.4f}")
    print(f"Model B - Accuracy: {acc_B:.4f}, Precision: {precision_B:.4f}, Recall: {recall_B:.4f}, F1 Score: {f1_B:.4f}")
    print(f"Model C - Accuracy: {acc_C:.4f}, Precision: {precision_C:.4f}, Recall: {recall_C:.4f}, F1 Score: {f1_C:.4f}")

df_scores = pd.DataFrame(scores)
df_scores.to_csv('../checkpoints/scores.csv', index=False)
df_scores.groupby('model').agg(['mean', 'std']).to_csv('../checkpoints/scores_summary.csv')
df_scores.groupby('model').agg(['mean', 'std']).to_clipboard()
# Paired t-test
f1_scores_model_A = [f1_score(all_ground_truth[i*split_size:(i+1)*split_size], all_preds_model_A[i*split_size:(i+1)*split_size]) for i in range(10)]
f1_scores_model_B = [f1_score(all_ground_truth[i*split_size:(i+1)*split_size], all_preds_model_B[i*split_size:(i+1)*split_size]) for i in range(10)]

t_stat, p_value = stats.ttest_rel(f1_scores_model_A, f1_scores_model_B)

print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There's a significant difference in the performance of the two models.")
else:
    print("There's no significant difference in the performance of the two models.")
