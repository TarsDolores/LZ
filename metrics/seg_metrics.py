import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        # super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

    @staticmethod
    def to_str(results, *args, **kwargs):
        # string = "\n"
        # for k, v in results.items():
        #     if k != "Class IoU" and k != "Class F1":  # 不打印类别相关的 IoU 和 F1
        #         string += "%s: %f\n" % (k, v)
        # return string
        string = "\n"

        # 打印 Overall Acc, Mean IoU 和 Avg F1
        string += "Overall Acc: %f\n" % results['Overall Acc']
        string += "Mean IoU: %f\n" % results['Mean IoU']
        string += "Avg F1: %f\n" % results['Avg F1']

        # 打印 F1 值，每个类别在同一行
        string += "F1:\n"
        string += " ".join([f"{results['Class F1'].get(i, 0):.4f}" for i in range(6)]) + "\n"

        # 打印 IoU 值，每个类别在同一行
        string += "IoU:\n"
        string += " ".join([f"{results['Class IoU'].get(i, 0):.4f}" for i in range(6)]) + "\n"

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self,exclude_classes = [3]):
        hist = self.confusion_matrix
        if exclude_classes is not None:
            # 排除指定类别的行和列
            valid_classes = [i for i in range(self.n_classes) if i not in exclude_classes]
            acc = np.diag(hist[valid_classes, :][:, valid_classes]).sum() / hist[valid_classes, :].sum()
        else:
            acc = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        cls_iu = dict(zip(range(self.n_classes), iu))
        if exclude_classes is not None:
            iu = np.delete(iu, exclude_classes)  #
        mean_iu = np.nanmean(iu)
        f1_scores = {}
        for i in range(self.n_classes):
            if exclude_classes is not None and i in exclude_classes:
                continue
            tp = hist[i, i]
            fp = np.sum(hist[:, i]) - tp
            fn = np.sum(hist[i, :]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[i] = f1  # **新增：将每个类别的 F1 值记录到字典中**
        avg_f1 = np.mean(list(f1_scores.values()))
        return {
                "Overall Acc": acc,
                "Mean IoU": mean_iu,
                "Avg F1": avg_f1,  #
                "Class IoU": cls_iu,
                "Class F1": f1_scores,  # **新增：每类 F1 值输出**
            }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """ Computes average values """
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
