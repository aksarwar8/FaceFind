
import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two face embeddings.
    """

    a = np.asarray(embedding_a, dtype=np.float32).flatten()
    b = np.asarray(embedding_b, dtype=np.float32).flatten()

    if a.shape != b.shape:
        raise ValueError("Embedding dimensions must match.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero-length embedding is invalid.")

    similarity = np.dot(a, b) / (norm_a * norm_b)

    return float(np.clip(similarity, -1.0, 1.0))
