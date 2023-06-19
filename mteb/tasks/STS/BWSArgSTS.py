import datasets

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
        
    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], 
            revision=self.description.get("revision", None)
        )

        # add the topic to the start of each sentence in the pair
        self.dataset = self.dataset.map(
            lambda x: {
                "sent1": f"[Topic: {x['topic'].lower()}] {x['sent1']}",
                "sent2": f"[Topic: {x['topic'].lower()}] {x['sent2']}",
            }
        )

        self.data_loaded = True