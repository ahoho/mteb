import datasets

from ...abstasks.AbsTaskArgSTS import AbsTaskArgSTS


class ArgFacetSTS(AbsTaskArgSTS):
    @property
    def description(self):
        return {
            "name": "ArgFacetSTS",
            "hf_hub_name": "./data/pairwise_files/arg_facet_sim",  # path is relative to evaluation script
            "description": "Argument Facet Similarity Corpus",
            "reference": "https://nlds.soe.ucsc.edu/node/44",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["val", "test"],
            "eval_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }