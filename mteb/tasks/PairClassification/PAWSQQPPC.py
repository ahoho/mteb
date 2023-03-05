from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PAWSQQPPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "PAWSQQPPC",
            "hf_hub_name": "./data/paws_qqp/data_qqp/paws_qqp/jsonl", # path is relative to evaluation script
            "description": "PAWS Adversarial Paraphrase Dataset, Quora Question Pairs",
            "reference": "https://github.com/google-research-datasets/paws",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train", "test"],
            "eval_langs": ["en"],
            "main_score": "ap",
        }