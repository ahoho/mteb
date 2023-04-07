import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from ...abstasks.AbsTaskClustering import AbsTaskClustering


class BillClusteringP2P(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "BillClusteringP2P",
            "hf_hub_name": "./data/bills",
            "description": (
                "Clustering of bill summaries in congress"
            ),
            "reference": "https://www.comparativeagendas.net/",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["train", "test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
        }
    

    def read_jsonl(self, fpath, label_key="topic", sublabel_key="subtopic", text_key="summary"):
        data = {"sentences": [], "labels": [], "sublabels": []}
        with open(fpath) as infile:
            for line in infile:
                if line:
                    line_data = json.loads(line)
                    data["sentences"].append(line_data[text_key])
                    data["labels"].append(line_data[label_key])
                    data["sublabels"].append(line_data[sublabel_key])
        return data


    def load_data(self, **kwargs):
        """
        Load dataset
        """
        if self.data_loaded:
            return

        # load jsonlines dataset, keeping only the `text` and `label` keys
        train_data = self.read_jsonl(Path(self.description["hf_hub_name"], "train.metadata.jsonl"))
        test_data = self.read_jsonl(Path(self.description["hf_hub_name"], "test.metadata.jsonl"))
        
        # match the strange format of the other cluster tasks
        self.dataset = DatasetDict(
            {
                "test": Dataset.from_list(
                    [train_data, test_data],
                )
            }
        )
        self.data_loaded = True