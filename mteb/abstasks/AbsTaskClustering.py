from collections import defaultdict

import numpy as np
import tqdm

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask


class AbsTaskClustering(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        metrics = defaultdict(list)
        for cluster_set in tqdm.tqdm(self.dataset[split], desc="Clustering"):
            evaluator = ClusteringEvaluator(cluster_set["sentences"], cluster_set["labels"], **kwargs)
            result = evaluator(model)
            for k, v in result.items():
                metrics[k].append(v)
        
        metric_summaries = {}
        for k, v in metrics.items():
            metric_summaries[k] = np.mean(v)
            metric_summaries[f"{k}_std"] = np.std(v)
        return metric_summaries
