from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SemEvalArgSimPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "SemEvalArgSimPC",
            "hf_hub_name": "./data/pairwise_files/semeval16_task6", # path is relative to evaluation script
            "description": "Tagged Arguments from argument similarity datasets.",
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
            self.dataset[split] = self.dataset[split].rename_column("score", "labels")