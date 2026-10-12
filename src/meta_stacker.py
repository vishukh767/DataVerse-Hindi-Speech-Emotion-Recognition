import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score


def compute_asr_quality_gate(asr_meta: np.ndarray, texts: list[str]) -> np.ndarray:
    """
    Computes a continuous reliability score in [0, 1] for each Whisper transcript
    based on log-probability confidence, no-speech probability, and word count.
    """
    word_counts = np.array([len(str(t).strip().split()) for t in texts], dtype=np.float32)
    avg_logprob = asr_meta[:, 0]
    no_speech = asr_meta[:, 1]
    conf = np.clip(np.exp(avg_logprob), 0.1, 1.0) * (1.0 - np.clip(no_speech, 0.0, 0.9))
    len_factor = np.clip(word_counts / 5.0, 0.15, 1.0)
    return np.clip(conf * len_factor, 0.10, 1.0)


def fit_class_specific_meta_stacker(
    oof_branches: list[np.ndarray],
    test_branches: list[np.ndarray],
    y_train: np.ndarray,
    skf_splits: list,
    non_twin_mask: np.ndarray,
):
    """
    Trains an out-of-fold L2-regularized class-specific meta-stacker on non-twin
    samples, combining linear probability blending with log-odds Product-of-Experts.
    """
    X_meta_tr = np.hstack([np.hstack(oof_branches), np.log(np.clip(np.hstack(oof_branches), 1e-4, 1.0))])
    X_meta_te = np.hstack([np.hstack(test_branches), np.log(np.clip(np.hstack(test_branches), 1e-4, 1.0))])

    n_samples = len(y_train)
    n_classes = oof_branches[0].shape[1]
    oof_meta = np.zeros((n_samples, n_classes), dtype=np.float64)
    test_meta = np.zeros((len(test_branches[0]), n_classes), dtype=np.float64)

    for tr_idx, va_idx in skf_splits:
        tr_clean = [i for i in tr_idx if non_twin_mask[i]]
        clf = LogisticRegression(C=0.15, max_iter=1000)
        clf.fit(X_meta_tr[tr_clean], y_train[tr_clean])
        oof_meta[va_idx] = clf.predict_proba(X_meta_tr[va_idx])
        test_meta += clf.predict_proba(X_meta_te) / len(skf_splits)

    return oof_meta, test_meta


def tune_class_multipliers(probs_oof: np.ndarray, y_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Coordinate-ascent + Nelder-Mead class prior temperature tuning on non-twin OOF.
    """
    best_m = np.ones(probs_oof.shape[1], dtype=np.float64)
    best_f1 = f1_score(y_true[mask], np.argmax(probs_oof[mask], axis=1), average="macro")

    def loss_func(m):
        scaled = probs_oof[mask] * m[None, :]
        return -f1_score(y_true[mask], np.argmax(scaled, axis=1), average="macro")

    res = minimize(loss_func, best_m, method="Nelder-Mead", options={"maxiter": 400})
    if -res.fun > best_f1 + 0.0015:
        return res.x / np.mean(res.x)
    return best_m
