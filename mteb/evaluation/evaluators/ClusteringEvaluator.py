import logging

import numpy as np
import sklearn
import sklearn.cluster


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


def get_majority_label(assignments):
    """
    Given a list of cluster assignments, return the majority label.
    """
    # assignments is (n_samples x n_assigned_labels). get most common label per row.
    # in cases with no majority, return the label in the first column
    majority_labels = []
    for row in assignments:
        unique, counts = np.unique(row, return_counts=True)
        if counts.max() == 1:
            majority_labels.append(row[0])
        else:
            majority_labels.append(unique[np.argmax(counts)])
    return majority_labels


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, clustering_batch_size=500, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences, labels = self._matched_random_sample(sentences, labels, size=limit)
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        corpus_embeddings = np.asarray(model.encode(self.sentences))
        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size
        )
        is_multi_embed_model = False
        if getattr(model, "output_combination_strategy", None) == "concat_embeds":
            is_multi_embed_model = True
            embed_dim = model.embedding_model.get_sentence_embedding_dimension()
            corpus_embeddings = corpus_embeddings.reshape(-1, embed_dim)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        if is_multi_embed_model:
            cluster_assignment = get_majority_label(cluster_assignment.reshape(len(self.labels), -1))

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)
        ari = sklearn.metrics.cluster.adjusted_rand_score(self.labels, cluster_assignment)
        ami = sklearn.metrics.cluster.adjusted_mutual_info_score(self.labels, cluster_assignment)

        return {
            "v_measure": v_measure,
            "adjusted_rand": ari,
            "adjusted_mutual_info": ami,
        }