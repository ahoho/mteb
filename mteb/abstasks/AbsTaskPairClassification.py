import logging
from collections import defaultdict

from ..evaluation.evaluators import PairClassificationEvaluator
from .AbsTask import AbsTask


class AbsTaskPairClassification(AbsTask):
    """
    Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()
        
        if len(self.dataset[split]) == 1: # artifact of strange way MTEB data is saved
            data_split = self.dataset[split][0]
        else:
            data_split = self.dataset[split]
        logging.getLogger("sentence_transformers.evaluation.PairClassificationEvaluator").setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sent1"], data_split["sent2"], data_split["labels"], **kwargs
        )
        scores = evaluator.compute_metrics(model)

        # Compute max
        max_scores = defaultdict(list)
        for sim_fct in scores:
            for metric in ["accuracy", "f1", "ap"]:
                max_scores[metric].append(scores[sim_fct][metric])

        for metric in max_scores:
            max_scores[metric] = max(max_scores[metric])

        scores["max"] = dict(max_scores)

        return scores
