from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification
import datasets

class PAWSWikiPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "PAWSWikiPC",
            "hf_hub_name": "paws", # path is relative to evaluation script
            "description": "PAWS Adversarial Paraphrase Dataset",
            "reference": "https://github.com/google-research-datasets/paws",
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

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], "labeled_final", **kwargs
        )
        self.dataset = self.dataset.rename_columns({"sentence1": "sent1", "sentence2": "sent2", "label": "labels"})
        self.data_loaded = True