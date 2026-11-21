
import numpy as np


class FaceMatcher:
    def __init__(
        self,
        recognition_threshold=0.50,
        possible_threshold=0.35,
        margin_threshold=0.05,
    ):
        self.recognition_threshold = recognition_threshold
        self.possible_threshold = possible_threshold
        self.margin_threshold = margin_threshold

    def match(self, query_embedding, gallery):
        names = gallery["names"]
        gallery_embeddings = gallery["embeddings"]

        if len(names) == 0:
            raise ValueError("Reference gallery is empty.")

        query = np.asarray(
            query_embedding, dtype=np.float32
        ).flatten()

        if gallery_embeddings.shape != (len(names), query.size):
            raise ValueError("Gallery embedding dimensions do not match.")

        query_norm = np.linalg.norm(query)
        gallery_norms = np.linalg.norm(
            gallery_embeddings, axis=1
        )

        if query_norm == 0 or np.any(gallery_norms == 0):
            raise ValueError("Zero-length embedding is invalid.")

        # Calculate cosine similarity against every reference identity.
        scores = (
            gallery_embeddings @ query
        ) / (gallery_norms * query_norm)

        scores = np.clip(scores, -1.0, 1.0)

        # Find the best and second-best candidates.
        order = np.argsort(scores)[::-1]

        best_index = int(order[0])
        best_score = float(scores[best_index])

        if len(order) > 1:
            second_score = float(scores[order[1]])
        else:
            second_score = -1.0

        margin = best_score - second_score

        # Apply open-set decision rules.
        if (
            best_score >= self.recognition_threshold
            and margin >= self.margin_threshold
        ):
            status = "recognised"
            identity = names[best_index]

        elif best_score >= self.possible_threshold:
            status = "possible"
            identity = names[best_index]

        else:
            status = "unknown"
            identity = None

        return {
            "identity": identity,
            "status": status,
            "best_score": best_score,
            "second_score": second_score,
            "margin": float(margin),
            "candidates": [
                {
                    "name": names[int(i)],
                    "score": float(scores[i]),
                }
                for i in order[:3]
            ],
        }
