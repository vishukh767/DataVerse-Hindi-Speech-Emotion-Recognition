from collections import defaultdict
import numpy as np


def compute_wave_signature(wav: np.ndarray, head_samples: int = 8000):
    """
    Returns (exact_num_samples, normalized_head_vector) for fast exact-length
    zero-lag cross-correlation twin matching.
    """
    n = len(wav)
    seg = wav[: min(n, head_samples)].astype(np.float32)
    seg = seg - np.mean(seg)
    norm = np.linalg.norm(seg) + 1e-12
    return n, seg / norm


def find_exact_waveform_twins(train_sigs, test_sigs, corr_threshold: float = 0.90):
    """
    Detects exact-length, phase-aligned duplicate recordings (r > corr_threshold)
    between train-train, train-test, and test-test splits.
    """
    len_to_train = defaultdict(list)
    for idx, (length, vec) in enumerate(train_sigs):
        len_to_train[length].append((idx, vec))

    train_test_twins = {}
    for te_idx, (te_len, te_vec) in enumerate(test_sigs):
        best_r, best_tr = -1.0, -1
        for tr_idx, tr_vec in len_to_train.get(te_len, []):
            m = min(len(te_vec), len(tr_vec))
            r = float(np.dot(te_vec[:m], tr_vec[:m]))
            if r > best_r:
                best_r, best_tr = r, tr_idx
        if best_r >= corr_threshold:
            train_test_twins[te_idx] = (best_tr, best_r)

    len_to_test = defaultdict(list)
    for te_idx, (te_len, te_vec) in enumerate(test_sigs):
        len_to_test[te_len].append((te_idx, te_vec))

    test_test_pairs = []
    for length, items in len_to_test.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx1, v1 = items[i]
                idx2, v2 = items[j]
                m = min(len(v1), len(v2))
                r = float(np.dot(v1[:m], v2[:m]))
                if r >= corr_threshold:
                    test_test_pairs.append((idx1, idx2, r))

    return train_test_twins, test_test_pairs
