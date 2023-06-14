from ...abstasks.AbsTaskSTS import AbsTaskSTS


class BWSArgSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "BWSArgSTS",
            "hf_hub_name": "./data/pairwise_files/bws_arg_sim",  # path is relative to evaluation script
            "description": "Argument similarity dataset from the BWS Argument Similarity Corpus.",
            "reference": "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2496",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }

