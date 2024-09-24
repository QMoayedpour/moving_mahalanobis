import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score


def all_metrics(gt_list, scores, model="...", n_ano=0, n_data=0, verbose=False):
    acc, f1, pre, rec = get_metrics(gt_list, scores, verbose=verbose)
    auc, aupr, fpr95 = plot_roc(gt_list, scores, verbose=verbose)
    if n_ano == 0:
        n_ano = int(sum(gt_list))
    if n_data == 0:
        n_data = int(len(gt_list))
    return {"Accuracy": acc, "F1": f1, "Precision": pre,
            "Recall": rec, "Roc_Auc": auc, "AUPr": aupr,
            "FPR@95%": fpr95, "n_anomaly": n_ano,
            "n_data": n_data, "model": model}


def plot_roc(gt_list, img_scores, verbose=True):
    fpr, tpr, thresholds_roc = roc_curve(gt_list, img_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds_pr = precision_recall_curve(gt_list, img_scores)
    aupr = average_precision_score(gt_list, img_scores)

    tpr_target = 0.95
    fpr_at_95_tpr = np.interp(tpr_target, tpr, fpr)

    if verbose:
        print(f'AUC: {roc_auc:.4f}')
        print(f'AUPR: {aupr:.4f}')
        print(f'Taux de faux positifs à 95% de sensibilité : {fpr_at_95_tpr:.4f}')

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        desc = f'FPR at 95% TPR = {fpr_at_95_tpr:.2f}'
        plt.scatter(fpr_at_95_tpr, tpr_target, color='red', zorder=5, label=desc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        plt.figure()
        desc = f'Precision-Recall curve (AUPR = {aupr:.2f})'
        plt.plot(recall, precision, color='blue', lw=2, label=desc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
    return float(roc_auc), float(aupr), float(fpr_at_95_tpr)


def get_metrics(gt_list, img_scores, verbose=True):
    precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    predictions = [1 if score >= threshold else 0 for score in img_scores]
    accuracy = accuracy_score(gt_list, predictions)
    f1 = f1_score(gt_list, predictions)
    precision = precision_score(gt_list, predictions)
    recall = recall_score(gt_list, predictions)

    if verbose:
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
    return float(accuracy), float(f1), float(precision), float(recall)

