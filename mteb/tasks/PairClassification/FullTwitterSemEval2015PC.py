from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class FullTwitterSemEval2015PC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "FullTwitterSemEval2015PC",
            "hf_hub_name": "./data/SemEval-PIT2015-github", # path is relative to evaluation script
            "description": "Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
            "reference": "https://alt.qcri.org/semeval2015/task1/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train", "validation", "test"],
            "eval_langs": ["en"],
            "main_score": "ap",
            "revision": "70970daeab8776df92f5ea462b6173c0b46fd2d1",
        }

    def load_data(self, **kwargs):
        super().load_data(**kwargs)
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].filter(lambda ex: ex["labels"] is not None)