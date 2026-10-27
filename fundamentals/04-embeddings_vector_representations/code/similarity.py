import numpy as np

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def dot_product(a, b):
    return float(np.dot(a, b))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

def binary_quantize(v):
    return (np.asarray(v) > 0).astype(np.uint8)


def hamming_distance(a, b):
    return int(np.count_nonzero(binary_quantize(a) != binary_quantize(b)))
