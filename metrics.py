import numpy as np

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return (a.T @ b) / (na * nb)

def kl_div(p,q):
    p = softmax(p)
    q = softmax(q) + 1e-15
    return np.sum(p * np.log(p/q))


def jensen_shannon_div(p,q):
    p = softmax(p*1e-3) + 1e-15
    q = softmax(q*1e-3) + 1e-15
    m = 0.5 * (p + q)
    return 0.5*np.sum(p * np.log2(p/m)) + 0.5*np.sum(q * np.log2(q/m))