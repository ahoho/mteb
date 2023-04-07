from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification

import datasets

class QQPPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "QQP",
            "hf_hub_name": "glue/qqp",
            "description": "Quora Question Pairs",
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train", "validation", "test"],
            "eval_langs": ["en"],
            "main_score": "ap",
        }
    
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset("glue", "qqp")
        self.dataset = self.dataset.rename_columns({"question1": "sent1", "question2": "sent2", "label": "labels"})
        self.data_loaded = True