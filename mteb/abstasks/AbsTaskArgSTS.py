from ..evaluation.evaluators import ArgSTSEvaluator
from .AbsTask import AbsTask


class AbsTaskArgSTS(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.description["min_score"]

    @property
    def max_score(self):
        return self.description["max_score"]

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores
    
    def _evaluate_split(self, model, data_split, **kwargs):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        s1 = "sentence1" if "sentence1" in data_split.column_names else "sent1"
        s2 = "sentence2" if "sentence2" in data_split.column_names else "sent2"
        evaluator = ArgSTSEvaluator(data_split[s1], data_split[s2], data_split["topic"], normalized_scores, **kwargs)
        metrics = evaluator(model)
        return metrics
