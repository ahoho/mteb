import datasets

from ...abstasks.AbsTaskArgSTS import AbsTaskArgSTS


class UKPArgAspectSTS(AbsTaskArgSTS):
    @property
    def description(self):
        return {
            "name": "UKPArgAspectSTS",
            "hf_hub_name": "./data/pairwise_files/ukp_aspect",  # path is relative to evaluation script
            "description": "UKP Argument Aspect Similarity Corpus",
            "reference": "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/1998",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 2,
        }