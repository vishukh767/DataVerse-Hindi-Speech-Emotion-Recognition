import numpy as np
import soundfile as sf
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torch
        t_wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(t_wav, sr, target_sr).squeeze(0).numpy()
    return wav.astype(np.float32)


def extract_67_prosody_features(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts 67 raw acoustic & prosodic features per utterance:
      - Duration, overall RMS energy, peak amplitude, dynamic range, crest factor
      - Frame-level RMS energy contours (mean, std, max, p10, p50, p90, delta, skewness)
      - Zero-Crossing Rate (ZCR) statistics
      - Voiced/Unvoiced ratio & Autocorrelation Fundamental Frequency (F0 / Pitch in Hz)
      - 7-band STFT log-power spectral distribution & spectral flux
      - Temporal thirds (beginning / middle / end energy progression)
    """
    if len(wav) < 400:
        wav = np.pad(wav, (0, 400 - len(wav)))

    duration = len(wav) / float(sr)
    rms_all = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
    peak = float(np.max(np.abs(wav)) + 1e-12)
    crest = float(peak / rms_all)

    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = max(1, 1 + (len(wav) - frame_len) // hop_len)
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n_frames, frame_len),
        strides=(wav.strides[0] * hop_len, wav.strides[0]),
    )
    window = np.hanning(frame_len).astype(np.float32)
    w_frames = frames * window

    frame_rms = np.sqrt(np.mean(w_frames ** 2, axis=1) + 1e-12)
    log_rms = np.log(frame_rms + 1e-8)
    d_rms = np.diff(log_rms, prepend=log_rms[0])

    signs = np.sign(frames)
    zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

    energy_thresh = np.percentile(frame_rms, 30)
    voiced_mask = frame_rms > energy_thresh
    voiced_ratio = float(np.mean(voiced_mask))

    min_lag = int(sr / 450.0)
    max_lag = int(sr / 70.0)
    f0_vals = []
    for i in range(0, n_frames, 2):
        if not voiced_mask[i]:
            continue
        f = w_frames[i]
        f = f - np.mean(f)
        norm = np.sum(f ** 2) + 1e-12
        ac = np.correlate(f, f, mode="full")[frame_len - 1 :] / norm
        if len(ac) > max_lag:
            sub = ac[min_lag:max_lag]
            if len(sub) > 0:
                pk_idx = int(np.argmax(sub))
                if sub[pk_idx] > 0.25:
                    f0_vals.append(sr / float(min_lag + pk_idx))

    if len(f0_vals) >= 3:
        f0_arr = np.array(f0_vals, dtype=np.float32)
        f0_feats = [
            float(np.mean(f0_arr)),
            float(np.std(f0_arr)),
            float(np.max(f0_arr)),
            float(np.min(f0_arr)),
            float(np.percentile(f0_arr, 10)),
            float(np.percentile(f0_arr, 50)),
            float(np.percentile(f0_arr, 90)),
            float(np.percentile(f0_arr, 90) - np.percentile(f0_arr, 10)),
            float(np.mean(np.abs(np.diff(f0_arr)))),
            float((f0_arr[-1] - f0_arr[0]) / max(1, len(f0_arr))),
        ]
    else:
        f0_feats = [0.0] * 10

    spec = np.abs(np.fft.rfft(w_frames, n=512, axis=1)) ** 2 + 1e-10
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)
    spec_norm = spec / (np.sum(spec, axis=1, keepdims=True) + 1e-10)
    centroid = np.sum(spec_norm * freqs[None, :], axis=1)
    spread = np.sqrt(np.sum(spec_norm * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) + 1e-10)
    flux = np.sqrt(np.mean(np.diff( np.log(spec), axis=0, prepend=np.log(spec[:1])) ** 2, axis=1))

    bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5500), (5500, 8000)]
    band_feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        b_energy = np.log(np.mean(spec[:, mask], axis=1) + 1e-10)
        band_feats.extend([
            float(np.mean(b_energy)),
            float(np.std(b_energy)),
            float(np.percentile(b_energy, 90) - np.percentile(b_energy, 10)),
            float(np.max(b_energy)),
        ])

    t1 = max(1, n_frames // 3)
    t2 = max(2, (2 * n_frames) // 3)
    e_start = float(np.mean(log_rms[:t1]))
    e_mid = float(np.mean(log_rms[t1:t2]))
    e_end = float(np.mean(log_rms[t2:]))

    feats = [
        duration,
        np.log(duration + 1e-4),
        rms_all,
        peak,
        crest,
        voiced_ratio,
        float(np.mean(log_rms)),
        float(np.std(log_rms)),
        float(np.max(log_rms)),
        float(np.min(log_rms)),
        float(np.percentile(log_rms, 10)),
        float(np.percentile(log_rms, 50)),
        float(np.percentile(log_rms, 90)),
        float(np.percentile(log_rms, 90) - np.percentile(log_rms, 10)),
        float(np.mean(np.abs(d_rms))),
        float(np.std(d_rms)),
        float(np.max(d_rms)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.max(zcr)),
        float(np.percentile(zcr, 90) - np.percentile(zcr, 10)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.percentile(centroid, 90) - np.percentile(centroid, 10)),
        float(np.mean(spread)),
        float(np.std(spread)),
        float(np.mean(flux)),
        float(np.std(flux)),
        float(np.max(flux)),
        e_start,
        e_mid,
        e_end,
        e_end - e_start,
        e_mid - 0.5 * (e_start + e_end),
    ]
    feats.extend(f0_feats)
    feats.extend(band_feats)
    return np.nan_to_num(np.array(feats[:67], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
