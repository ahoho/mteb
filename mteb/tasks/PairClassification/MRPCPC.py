from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification

import datasets

class MRPCPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "MRPC",
            "hf_hub_name": "glue/mrpc", # path is relative to evaluation script
            "description": "Microsoft research paraphrase corpus",
            "reference": "https://www.microsoft.com/en-us/download/details.aspx?id=52398",
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

        self.dataset = datasets.load_dataset("glue", "mrpc")
        self.dataset = self.dataset.rename_columns({"sentence1": "sent1", "sentence2": "sent2", "label": "labels"})
        self.data_loaded = True