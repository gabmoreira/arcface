import numpy as np


def cosineSimilarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return (a @ b) / (na * nb)