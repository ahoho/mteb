from ...abstasks.AbsTaskSTS import AbsTaskSTS


class FullTwitterSemEval2015STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "FullTwitterSemEval2015STS",
            "hf_hub_name": "./data/SemEval-PIT2015-github",  # path is relative to evaluation script
            "description": "Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
            "reference": "https://alt.qcri.org/semeval2015/task1/",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["train", "dev", "test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "ae752c7c21bf194d8b67fd573edf7ae58183cbe3",
        }
