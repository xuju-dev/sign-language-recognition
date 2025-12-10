import numpy as np
from scipy.stats import friedmanchisquare
from evaluate import load as load_metric

def get_classification_metrics():
    """
    Returns a compute_metrics function for classification tasks.
    Supports accuracy and weighted F1 by default.
    """
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    conf_matrix_metric = load_metric("confusion_matrix")

    def compute_metrics(eval_pred):
        """
        Hugging Face Trainer expects this signature:
        eval_pred -> (predictions, labels)
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_metric.compute(predictions=preds, references=labels)
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        conf_matrix = conf_matrix_metric.compute(predictions=preds, references=labels)

        return {
            "accuracy": acc["accuracy"],
            "macro_f1": macro_f1["f1"],
            "confusion_matrix": conf_matrix["confusion_matrix"]
        }

    return compute_metrics

def statistical_analysis(*samples):
    """
    Perform Friedman test on multiple samples.
    Each sample is a list of metric values from different runs.
    Returns the test statistic and p-value.
    """
    stat, p = friedmanchisquare(*samples)

    # TODO: Wilcoxon paired test

    # TODO: Hommel post-hoc test
    return stat, p