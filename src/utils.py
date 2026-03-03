from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence, TypedDict

import numpy as np
import yaml


class AppConfig(TypedDict):
    project_root: Path
    data_dir: Path
    recording_dir_name: str
    recording_name: str
    stem: str
    window_s: float | None
    channel: int


ELECTRODE_TO_CHANNEL: dict[str, int] = {
    "88": 2, "78": 1, "68": 3, "58": 4, "48": 6, "38": 8, "28": 10, "18": 14,
    "96": 65, "87": 66, "77": 33, "67": 34, "57": 7, "47": 9, "37": 11, "27": 12, "17": 16, "08": 18,
    "95": 67, "86": 68, "76": 35, "66": 36, "56": 5, "46": 17, "36": 13, "26": 23, "16": 20, "07": 22,
    "94": 69, "85": 70, "75": 37, "65": 38, "55": 48, "45": 15, "35": 19, "25": 25, "15": 27, "06": 24,
    "93": 71, "84": 72, "74": 39, "64": 40, "54": 42, "44": 50, "34": 54, "24": 21, "14": 29, "05": 26,
    "92": 73, "83": 74, "73": 41, "63": 43, "53": 44, "43": 46, "33": 52, "23": 62, "13": 31, "04": 28,
    "91": 75, "82": 76, "72": 45, "62": 47, "52": 51, "42": 56, "32": 58, "22": 60, "12": 64, "03": 30,
    "90": 77, "81": 78, "71": 82, "61": 49, "51": 53, "41": 55, "31": 57, "21": 59, "11": 61, "02": 32,
    "89": 79, "80": 80, "70": 84, "60": 86, "50": 87, "40": 89, "30": 91, "20": 94, "10": 63, "01": 95,
    "79": 81, "69": 83, "59": 85, "49": 88, "39": 90, "29": 92, "19": 93, "09": 96,
}

CHANNEL_TO_ELECTRODE: dict[int, str] = {channel: electrode for electrode, channel in ELECTRODE_TO_CHANNEL.items()}


def find_project_root() -> Path:
    """Find repo root so file paths work from interactive and script sessions."""
    search_starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]

    for start in search_starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate

    raise FileNotFoundError("Could not locate project root (missing pyproject.toml).")


def load_yaml_config(config_path: Path | None = None) -> dict:
    project_root = find_project_root()
    yaml_path = config_path or (project_root / "config" / "config.yaml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing config file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {yaml_path}")

    return config


def load_app_config(config_path: Path | None = None) -> AppConfig:
    project_root = find_project_root()
    raw = load_yaml_config(config_path)

    paths = raw.get("paths", {})
    defaults = raw.get("defaults", {})

    data_root = Path(paths.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = project_root / data_root

    recording_dir = str(paths.get("recording_dir", "")).strip()
    data_dir = data_root / recording_dir if recording_dir else data_root

    window_value = defaults.get("window_s", None)
    window_s: float | None
    if window_value is None:
        window_s = None
    else:
        window_s = float(window_value)

    recording_name = str(defaults.get("recording_name", defaults.get("stem", "spontaneous_initial0018")))

    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "recording_dir_name": recording_dir if recording_dir else data_dir.name,
        "recording_name": recording_name,
        # Backward compatibility for older scripts that still reference `stem`.
        "stem": recording_name,
        "window_s": window_s,
        "channel": int(defaults.get("channel", 0)),
    }


def channel_to_electrode_label(channel_id: int) -> str | None:
    return CHANNEL_TO_ELECTRODE.get(int(channel_id))


def electrode_label_to_channel(electrode_label: str) -> int | None:
    return ELECTRODE_TO_CHANNEL.get(str(electrode_label))


def electrode_label_to_grid_index(electrode_label: str) -> tuple[int, int]:
    """
    Convert a two-digit electrode id ('01'..'96') to a 10x10 Utah grid index
    with only the four corners unused:
    - top row:    [corner, 01..08, corner]
    - middle rows [09..88] in row-major blocks of 10
    - bottom row: [corner, 89..96, corner]
    """
    if len(electrode_label) != 2 or not electrode_label.isdigit():
        raise ValueError(f"Invalid electrode label: {electrode_label}")
    electrode_id = int(electrode_label)
    if electrode_id < 1 or electrode_id > 96:
        raise ValueError(f"Electrode id out of range 01..96: {electrode_label}")

    if electrode_id <= 8:
        return 0, electrode_id
    if electrode_id <= 88:
        idx = electrode_id - 9
        return 1 + (idx // 10), idx % 10

    return 9, electrode_id - 88


def channels_to_utah_grid(values_by_channel: Mapping[int, float], fill_value: float = np.nan) -> np.ndarray:
    """
    Map channel-keyed values to a 10x10 Utah-array-style grid using ELECTRODE_TO_CHANNEL.
    Grid cells without values remain fill_value.
    """
    grid = np.full((10, 10), fill_value, dtype=float)
    for channel_id, value in values_by_channel.items():
        electrode_label = channel_to_electrode_label(int(channel_id))
        if electrode_label is None:
            continue
        row, col = electrode_label_to_grid_index(electrode_label)
        grid[row, col] = float(value)
    return grid


def utah_channel_grid(fill_value: float = np.nan) -> np.ndarray:
    """Return a 10x10 grid with channel ids in valid Utah electrode positions."""
    values = {int(ch): int(ch) for ch in CHANNEL_TO_ELECTRODE}
    return channels_to_utah_grid(values, fill_value=fill_value)


def utah_electrode_grid(fill_value: str = "") -> np.ndarray:
    """Return a 10x10 grid with electrode labels in valid Utah positions."""
    grid = np.full((10, 10), fill_value, dtype=object)
    for electrode_id in range(1, 97):
        electrode_label = f"{electrode_id:02d}"
        row, col = electrode_label_to_grid_index(electrode_label)
        grid[row, col] = electrode_label
    return grid


def nsx_extended_headers_by_channel(extended_headers: list[dict]) -> dict[int, dict]:
    """Index NSx extended headers by channel/electrode id."""
    out: dict[int, dict] = {}
    for header in extended_headers:
        if "ElectrodeID" in header:
            out[int(header["ElectrodeID"])] = header
    return out


def nsx_uV_per_bit_from_header(header: Mapping[str, object]) -> float:
    """
    Compute NSx conversion factor from raw int16 units to physical units.
    In Blackrock files this is typically uV/bit.
    """
    min_d = float(header["MinDigitalValue"])
    max_d = float(header["MaxDigitalValue"])
    min_a = float(header["MinAnalogValue"])
    max_a = float(header["MaxAnalogValue"])
    return (max_a - min_a) / (max_d - min_d)


def nev_waveform_bits_to_uV(waveforms: np.ndarray) -> np.ndarray:
    """
    Convert NEV waveform values from bits to microvolts.
    IFU states waveforms are in bits with 0.25 uV/bit.
    """
    return np.asarray(waveforms, dtype=np.float64) * 0.25


def _cluster_mean_waveforms_for_channel(
    channel_id: int,
    waveforms_uV: np.ndarray,
    *,
    max_clusters: int,
    pca_components: int,
    max_spikes_per_channel: int,
    min_spikes_for_clustering: int,
    min_cluster_size: int,
    min_cluster_fraction: float,
    random_state: int,
) -> tuple[int, list[np.ndarray]]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    n_spikes, n_samples = waveforms_uV.shape
    if n_spikes == 0:
        return channel_id, []

    rng = np.random.default_rng(random_state + int(channel_id))
    if n_spikes > max_spikes_per_channel:
        keep_idx = rng.choice(n_spikes, size=max_spikes_per_channel, replace=False)
        wf = waveforms_uV[keep_idx]
    else:
        wf = waveforms_uV

    # Remove waveform DC offset before PCA and means.
    wf = wf - wf.mean(axis=1, keepdims=True)
    n_used = wf.shape[0]
    if n_used == 0:
        return channel_id, []

    labels = np.zeros(n_used, dtype=np.int32)
    if n_used >= min_spikes_for_clustering and max_clusters > 1:
        n_comp = min(pca_components, n_used, n_samples)
        if n_comp >= 1:
            z = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state).fit_transform(wf)
            max_k_by_size = max(1, n_used // max(min_cluster_size, 1))
            n_clusters = min(max_clusters, max_k_by_size)
            if n_clusters > 1:
                km = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    n_init="auto",
                    batch_size=min(512, n_used),
                    random_state=random_state,
                )
                labels = km.fit_predict(z)

    keep_threshold = max(min_cluster_size, int(np.ceil(min_cluster_fraction * n_used)))
    cluster_means: list[tuple[int, np.ndarray]] = []
    for cid in sorted(np.unique(labels)):
        cid_mask = labels == cid
        cid_count = int(np.count_nonzero(cid_mask))
        if cid_count < keep_threshold:
            continue
        mw = wf[cid_mask].mean(axis=0)
        mw = mw - mw.mean()
        cluster_means.append((cid_count, mw))

    if not cluster_means:
        mw = wf.mean(axis=0)
        mw = mw - mw.mean()
        return channel_id, [mw]

    cluster_means.sort(key=lambda x: x[0], reverse=True)
    return channel_id, [mw for _, mw in cluster_means]


def pca_cluster_mean_waveforms_by_channel(
    waveforms_uV: np.ndarray,
    channels: np.ndarray,
    *,
    max_clusters: int = 3,
    pca_components: int = 3,
    max_spikes_per_channel: int = 5000,
    min_spikes_for_clustering: int = 80,
    min_cluster_size: int = 20,
    min_cluster_fraction: float = 0.05,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> dict[int, list[np.ndarray]]:
    """
    Fast channel-wise spike sorting surrogate:
    PCA + MiniBatchKMeans per channel, returning cluster mean waveforms in uV.
    """
    wf = np.asarray(waveforms_uV, dtype=np.float64)
    ch = np.asarray(channels, dtype=np.int64)

    if wf.ndim != 2:
        raise ValueError(f"waveforms_uV must have shape (n_spikes, n_samples), got {wf.shape}")
    if ch.ndim != 1:
        raise ValueError(f"channels must have shape (n_spikes,), got {ch.shape}")
    if wf.shape[0] != ch.size:
        raise ValueError(f"waveforms/channels length mismatch: {wf.shape[0]} vs {ch.size}")
    if wf.shape[0] == 0:
        return {}

    order = np.argsort(ch, kind="stable")
    wf_sorted = wf[order]
    ch_sorted = ch[order]
    unique_ch, starts, counts = np.unique(ch_sorted, return_index=True, return_counts=True)

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, min(8, os.cpu_count() or 1))

    out: dict[int, list[np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = []
        for channel_id, start, count in zip(unique_ch.tolist(), starts.tolist(), counts.tolist()):
            stop = start + count
            futures.append(
                pool.submit(
                    _cluster_mean_waveforms_for_channel,
                    int(channel_id),
                    wf_sorted[start:stop],
                    max_clusters=max_clusters,
                    pca_components=pca_components,
                    max_spikes_per_channel=max_spikes_per_channel,
                    min_spikes_for_clustering=min_spikes_for_clustering,
                    min_cluster_size=min_cluster_size,
                    min_cluster_fraction=min_cluster_fraction,
                    random_state=random_state,
                )
            )
        for future in futures:
            channel_id, means = future.result()
            if means:
                out[channel_id] = means

    return out


BANDPASS_RANGES_HZ: dict[str, tuple[float, float]] = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "high_gamma": (80.0, 200.0),
}


def resolve_bandpass_range_hz(band_name: str) -> tuple[float, float]:
    key = str(band_name).strip().lower().replace(" ", "_")
    if key not in BANDPASS_RANGES_HZ:
        valid = ", ".join(sorted(BANDPASS_RANGES_HZ))
        raise ValueError(f"Unknown band '{band_name}'. Valid bands: {valid}")
    return BANDPASS_RANGES_HZ[key]


def bandpass_filter_matrix(signals: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    Bandpass-filter one or many traces (last axis = time).
    Uses scipy zero-phase filtering when available, else FFT masking fallback.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
        squeeze = True
    elif x.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"signals must be 1D or 2D, got shape {x.shape}")

    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    nyquist = 0.5 * float(fs_hz)
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(f"Band must satisfy 0 < low < high < Nyquist ({nyquist:.3f} Hz)")

    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
        y = sosfiltfilt(sos, x, axis=-1)
    except Exception:
        # Fallback when scipy is unavailable: ideal FFT-domain bandpass.
        x_centered = x - x.mean(axis=-1, keepdims=True)
        n = x_centered.shape[-1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        mask = ((freqs >= low_hz) & (freqs <= high_hz))[np.newaxis, :]
        x_fft = np.fft.rfft(x_centered, axis=-1)
        y = np.fft.irfft(x_fft * mask, n=n, axis=-1)

    if squeeze:
        return y[0]
    return y


def slugify_name(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def create_results_dirs(project_root: Path, recording_dir_name: str, recording_name: str) -> dict[str, Path]:
    base_dir = Path(project_root) / "results" / str(recording_dir_name) / str(recording_name)
    plots_dir = base_dir / "plots"
    videos_dir = base_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return {"base_dir": base_dir, "plots_dir": plots_dir, "videos_dir": videos_dir}


def should_write_output(path: Path, ask_before_overwrite: bool) -> bool:
    if not path.exists():
        return True
    if not ask_before_overwrite:
        return True
    while True:
        try:
            answer = input(f"File exists: {path.name}. Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            print("Non-interactive session detected; keeping existing file:", path)
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            print("Keeping existing file:", path)
            return False


def save_figure(
    fig,
    plots_dir: Path,
    filename: str,
    *,
    ask_before_overwrite: bool,
    dpi: int = 220,
) -> Path | None:
    path = Path(plots_dir) / filename
    if not should_write_output(path, ask_before_overwrite=ask_before_overwrite):
        return None
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Saved plot:", path)
    return path


def _robust_log_ylim(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.8) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1e-12, 1.0
    y_min = float(np.percentile(vals, lo_pct))
    y_max = float(np.percentile(vals, hi_pct))
    if y_max <= y_min:
        y_min = float(np.min(vals))
        y_max = float(np.max(vals))
    return max(y_min * 0.9, 1e-12), y_max * 1.1


def plot_utah_layout(layout_channels: np.ndarray, layout_electrodes: np.ndarray, *, figsize: tuple[float, float] = (10, 10)):
    import matplotlib.pyplot as plt

    layout_mask = np.where(np.isnan(layout_channels), np.nan, 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(layout_mask, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    for r in range(layout_channels.shape[0]):
        for c in range(layout_channels.shape[1]):
            if np.isnan(layout_channels[r, c]):
                continue
            ch = int(layout_channels[r, c])
            elec = layout_electrodes[r, c]
            ax.text(c, r, f"E{elec}\nC{ch}", ha="center", va="center", fontsize=19, color="#e3b110")
    ax.set_title("Utah layout: electrode and channel labels")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_nev_waveforms(
    wf_uV: np.ndarray,
    waveform_channel: int,
    *,
    view_mode: str = "cluster",
    max_clusters_to_show: int = 4,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(wf_uV.shape[1])

    if view_mode == "cluster":
        from sklearn.decomposition import PCA
        import hdbscan

        z = PCA(n_components=3, random_state=0).fit_transform(wf_uV)
        labels = hdbscan.HDBSCAN(min_cluster_size=max(20, wf_uV.shape[0] // 20)).fit_predict(z)
        cluster_ids = [cid for cid in sorted(np.unique(labels)) if cid >= 0]
        if not cluster_ids:
            cluster_ids = [0]
            labels = np.zeros(wf_uV.shape[0], dtype=int)
        colors = ["black", "blue", "green", "red"]
        for unit_idx, cid in enumerate(cluster_ids[:max_clusters_to_show]):
            wf_c = wf_uV[labels == cid]
            if wf_c.size == 0:
                continue
            ax.plot(t, wf_c.T, color=colors[unit_idx % len(colors)], alpha=0.05, lw=0.7)
            ax.plot(
                t,
                wf_c.mean(axis=0),
                color=colors[unit_idx % len(colors)],
                lw=2.0,
                label=f"U{unit_idx + 1} n={wf_c.shape[0]}",
            )
        ax.legend(frameon=False)
        title_suffix = "PCA+HDBSCAN units"
    elif view_mode == "all":
        ax.plot(t, wf_uV.T, color="black", alpha=0.08)
        ax.plot(t, wf_uV.mean(axis=0), color="red", lw=1.4, label="mean")
        ax.legend(frameon=False)
        title_suffix = "all waveforms"
    else:
        mean_wf = wf_uV.mean(axis=0)
        std_wf = wf_uV.std(axis=0)
        ax.plot(t, mean_wf, color="black", label="mean")
        ax.fill_between(t, mean_wf - std_wf, mean_wf + std_wf, color="gray", alpha=0.25, label="+-1 SD")
        ax.legend(frameon=False)
        title_suffix = "mean +- SD"

    ax.set_title(f"NEV waveforms (ch {waveform_channel}, {title_suffix})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (uV)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_utah_cluster_mean_waveforms(
    layout_channels: np.ndarray,
    layout_electrodes: np.ndarray,
    mean_waveforms_by_channel: Mapping[int, list[np.ndarray]],
    *,
    max_units_per_channel: int = 3,
    normalization: str = "per_site",
):
    import matplotlib.pyplot as plt

    global_peak = 0.0
    for mw_list in mean_waveforms_by_channel.values():
        for mw in mw_list:
            global_peak = max(global_peak, float(np.max(np.abs(mw))))
    if global_peak <= 0:
        global_peak = 1.0

    site_peak_by_channel: dict[int, float] = {}
    if normalization == "per_site":
        for ch, mw_list in mean_waveforms_by_channel.items():
            local_peak = 0.0
            for mw in mw_list:
                local_peak = max(local_peak, float(np.max(np.abs(mw))))
            site_peak_by_channel[ch] = max(local_peak, 1.0)

    fig, axes = plt.subplots(10, 10, figsize=(14, 14), dpi=200, sharex=True, sharey=True)
    t = np.arange(next(iter(mean_waveforms_by_channel.values()))[0].size) if mean_waveforms_by_channel else np.arange(52)
    unit_colors = ["black", "#1f77b4", "#d62728", "#2ca02c"]

    for r in range(10):
        for c in range(10):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            if np.isnan(layout_channels[r, c]):
                ax.set_facecolor("#efefef")
                continue
            ch = int(layout_channels[r, c])
            elec = layout_electrodes[r, c]
            ax.set_facecolor("white")
            channel_means = mean_waveforms_by_channel.get(ch, [])
            channel_scale = site_peak_by_channel.get(ch, global_peak)
            if channel_scale <= 0:
                channel_scale = 1.0
            for unit_idx, mw in enumerate(channel_means[:max_units_per_channel]):
                y = mw / channel_scale
                ax.plot(t, y, color=unit_colors[unit_idx % len(unit_colors)], lw=1.4)
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"E{elec} C{ch} U{len(channel_means)}", fontsize=9, color="black", pad=1.0)

    fig.suptitle(f"PCA+cluster mean NEV waveforms by Utah site ({normalization})", y=0.995)
    fig.tight_layout()
    return fig


def _window_indices(total_s: float, fs: float, start_s: float, end_s: float | None) -> tuple[int, int]:
    win_start_s = min(max(start_s, 0.0), total_s)
    win_end_s = total_s if end_s is None else min(end_s, total_s)
    if win_end_s <= win_start_s:
        raise ValueError("End time must be > start time.")
    i0 = int(np.floor(win_start_s * fs))
    i1 = int(np.ceil(win_end_s * fs))
    return i0, i1


def plot_continuous_traces_ns2_ns5(
    ns2_seg: np.ndarray,
    ns2_fs: float,
    ns2_channels: np.ndarray,
    ns2_uV_per_bit: np.ndarray,
    ns5_seg: np.ndarray,
    ns5_fs: float,
    ns5_channels: np.ndarray,
    ns5_uV_per_bit: np.ndarray,
    channels_to_plot: Sequence[int],
    *,
    start_s: float,
    end_s: float | None,
    title_label: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(channels_to_plot), 2, figsize=(16, 3.8 * len(channels_to_plot)), sharex=False)
    if len(channels_to_plot) == 1:
        axes = np.array([axes])

    ns2_total_s = ns2_seg.shape[1] / ns2_fs
    ns5_total_s = ns5_seg.shape[1] / ns5_fs
    ns2_i0, ns2_i1 = _window_indices(ns2_total_s, ns2_fs, start_s, end_s)
    ns5_i0, ns5_i1 = _window_indices(ns5_total_s, ns5_fs, start_s, end_s)

    for i, channel_id in enumerate(channels_to_plot):
        row_ns2 = int(np.where(ns2_channels == channel_id)[0][0])
        row_ns5 = int(np.where(ns5_channels == channel_id)[0][0])
        elec = channel_to_electrode_label(channel_id)

        y2 = ns2_seg[row_ns2, ns2_i0:ns2_i1].astype(np.float64) * ns2_uV_per_bit[row_ns2]
        t2 = np.arange(ns2_i0, ns2_i1) / ns2_fs
        axes[i, 0].plot(t2, y2, color="black")
        axes[i, 0].set_title(f"NS2 ch {channel_id} (elec {elec})")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude (uV)")
        axes[i, 0].spines["top"].set_visible(False)
        axes[i, 0].spines["right"].set_visible(False)

        y5 = ns5_seg[row_ns5, ns5_i0:ns5_i1].astype(np.float64) * ns5_uV_per_bit[row_ns5]
        t5 = np.arange(ns5_i0, ns5_i1) / ns5_fs
        axes[i, 1].plot(t5, y5, color="black")
        axes[i, 1].set_title(f"NS5 ch {channel_id} (elec {elec})")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_ylabel("Amplitude (uV)")
        axes[i, 1].spines["top"].set_visible(False)
        axes[i, 1].spines["right"].set_visible(False)

    fig.suptitle(f"Continuous data in physical units: {title_label}", y=1.01)
    fig.tight_layout()
    return fig


def plot_fourier_power_ns2_ns5(
    ns2_seg: np.ndarray,
    ns2_fs: float,
    ns2_channels: np.ndarray,
    ns2_uV_per_bit: np.ndarray,
    ns5_seg: np.ndarray,
    ns5_fs: float,
    ns5_channels: np.ndarray,
    ns5_uV_per_bit: np.ndarray,
    channels_to_plot: Sequence[int],
    *,
    start_s: float,
    end_s: float | None,
    title_label: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(channels_to_plot), 2, figsize=(16, 3.8 * len(channels_to_plot)), sharex=False)
    if len(channels_to_plot) == 1:
        axes = np.array([axes])

    ns2_total_s = ns2_seg.shape[1] / ns2_fs
    ns5_total_s = ns5_seg.shape[1] / ns5_fs
    ns2_i0, ns2_i1 = _window_indices(ns2_total_s, ns2_fs, start_s, end_s)
    ns5_i0, ns5_i1 = _window_indices(ns5_total_s, ns5_fs, start_s, end_s)

    for i, channel_id in enumerate(channels_to_plot):
        row_ns2 = int(np.where(ns2_channels == channel_id)[0][0])
        row_ns5 = int(np.where(ns5_channels == channel_id)[0][0])
        elec = channel_to_electrode_label(channel_id)

        x2 = ns2_seg[row_ns2, ns2_i0:ns2_i1].astype(np.float64) * ns2_uV_per_bit[row_ns2]
        x2 = x2 - np.mean(x2)
        f2 = np.fft.rfftfreq(x2.size, d=1.0 / ns2_fs)
        p2 = (np.abs(np.fft.rfft(x2)) ** 2) / x2.size
        axes[i, 0].plot(f2, p2, color="black")
        axes[i, 0].set_xlim(0, 1000)
        axes[i, 0].axvline(ns2_fs / 2, color="gray", lw=0.8, ls="--")
        axes[i, 0].set_yscale("log")
        axes[i, 0].set_ylim(*_robust_log_ylim(p2))
        axes[i, 0].set_title(f"NS2 FFT power ch {channel_id} (elec {elec})")
        axes[i, 0].set_xlabel("Frequency (Hz)")
        axes[i, 0].set_ylabel("Power (uV^2)")
        axes[i, 0].spines["top"].set_visible(False)
        axes[i, 0].spines["right"].set_visible(False)

        x5 = ns5_seg[row_ns5, ns5_i0:ns5_i1].astype(np.float64) * ns5_uV_per_bit[row_ns5]
        x5 = x5 - np.mean(x5)
        f5 = np.fft.rfftfreq(x5.size, d=1.0 / ns5_fs)
        p5 = (np.abs(np.fft.rfft(x5)) ** 2) / x5.size
        axes[i, 1].plot(f5, p5, color="black")
        axes[i, 1].set_xlim(0, min(8000, ns5_fs / 2))
        axes[i, 1].set_yscale("log")
        axes[i, 1].set_ylim(*_robust_log_ylim(p5))
        axes[i, 1].set_title(f"NS5 FFT power ch {channel_id} (elec {elec})")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Power (uV^2)")
        axes[i, 1].spines["top"].set_visible(False)
        axes[i, 1].spines["right"].set_visible(False)

    fig.suptitle(f"Fourier power: {title_label}", y=1.01)
    fig.tight_layout()
    return fig


def plot_bandpass_traces(filtered_ns5_uV: np.ndarray, t_s: np.ndarray, channels: Sequence[int], *, band_label: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.4 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = np.array([axes])
    for i, ch in enumerate(channels):
        elec = channel_to_electrode_label(int(ch))
        axes[i].plot(t_s, filtered_ns5_uV[i], color="black", lw=0.8)
        axes[i].set_ylabel(f"ch {ch}\n(e{elec})")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"NS5 bandpass {band_label}", y=1.01)
    fig.tight_layout()
    return fig


def plot_filtered_band_power(
    raw_ns5_uV: np.ndarray,
    filtered_ns5_uV: np.ndarray,
    fs_hz: float,
    channels: Sequence[int],
    *,
    low_hz: float,
    high_hz: float,
    band_label: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.8 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = np.array([axes])

    max_display_hz = min(500.0, fs_hz / 2.0)
    for i, ch in enumerate(channels):
        ax = axes[i]
        x_raw = raw_ns5_uV[i] - np.mean(raw_ns5_uV[i])
        x_filt = filtered_ns5_uV[i] - np.mean(filtered_ns5_uV[i])
        freqs = np.fft.rfftfreq(x_raw.size, d=1.0 / fs_hz)
        p_raw = (np.abs(np.fft.rfft(x_raw)) ** 2) / x_raw.size
        p_filt = (np.abs(np.fft.rfft(x_filt)) ** 2) / x_filt.size
        ax.plot(freqs, p_raw, color="#aaaaaa", lw=0.8, label="raw")
        ax.plot(freqs, p_filt, color="black", lw=0.9, label="bandpass")
        ax.axvspan(low_hz, high_hz, color="#ffcf70", alpha=0.2)
        ax.set_yscale("log")
        mask = (freqs >= 0.0) & (freqs <= max_display_hz)
        ax.set_ylim(*_robust_log_ylim(np.concatenate([p_raw[mask], p_filt[mask]])))
        ax.set_ylabel(f"ch {ch}\nPower")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xlim(0, max_display_hz)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Filtered NS5 Fourier power ({band_label}: {low_hz:.0f}-{high_hz:.0f} Hz)", y=1.01)
    fig.tight_layout()
    return fig


def build_spike_rate_video_grids(
    spike_t_s: np.ndarray,
    spike_ch: np.ndarray,
    *,
    start_s: float,
    total_s: float,
    bin_ms: float,
    end_limit_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")
    bin_s = float(bin_ms) / 1000.0
    end_s = min(end_limit_s, start_s + total_s)
    if end_s <= start_s:
        raise ValueError("Invalid time range.")

    edges = np.arange(start_s, end_s + bin_s, bin_s)
    if edges[-1] < end_s:
        edges = np.append(edges, end_s)
    n_frames = edges.size - 1
    if n_frames <= 0:
        raise ValueError("No frames for the chosen settings.")

    frame_idx = np.digitize(spike_t_s, edges) - 1
    valid = (frame_idx >= 0) & (frame_idx < n_frames)
    frame_idx = frame_idx[valid]
    ch = spike_ch[valid]
    channels = np.array(sorted(set(ch.tolist())), dtype=int)
    ch_to_row = {int(c): i for i, c in enumerate(channels)}
    rows = np.array([ch_to_row[int(c)] for c in ch], dtype=int)
    flat_idx = rows * n_frames + frame_idx
    counts_flat = np.bincount(flat_idx, minlength=channels.size * n_frames)
    rate_matrix = counts_flat.reshape(channels.size, n_frames) / np.diff(edges)[None, :]

    grids = np.zeros((n_frames, 10, 10), dtype=float)
    for frame_i in range(n_frames):
        by_channel = {
            int(c): float(rate_matrix[row_i, frame_i])
            for row_i, c in enumerate(channels)
            if rate_matrix[row_i, frame_i] > 0
        }
        grids[frame_i] = np.nan_to_num(channels_to_utah_grid(by_channel), nan=0.0)
    return grids, edges


def build_band_power_video_grids(
    ns5_seg: np.ndarray,
    ns5_uV_per_bit: np.ndarray,
    ns5_channels: np.ndarray,
    ns5_fs: float,
    *,
    start_s: float,
    total_s: float,
    bin_ms: float,
    band_name: str,
    filter_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")
    low_hz, high_hz = resolve_bandpass_range_hz(band_name)
    bin_s = float(bin_ms) / 1000.0
    end_s = min(ns5_seg.shape[1] / ns5_fs, start_s + total_s)
    if end_s <= start_s:
        raise ValueError("Invalid time range.")

    edges = np.arange(start_s, end_s + bin_s, bin_s)
    if edges[-1] < end_s:
        edges = np.append(edges, end_s)
    n_frames = edges.size - 1
    if n_frames <= 0:
        raise ValueError("No frames for the chosen settings.")

    i0 = int(np.floor(start_s * ns5_fs))
    i1 = int(np.ceil(end_s * ns5_fs))
    n_samples = i1 - i0
    if n_samples <= 0:
        raise ValueError("Invalid NS5 sample window.")

    frame_starts = np.floor((edges[:-1] - start_s) * ns5_fs).astype(int)
    frame_ends = np.ceil((edges[1:] - start_s) * ns5_fs).astype(int)
    frame_starts = np.clip(frame_starts, 0, max(n_samples - 1, 0))
    frame_ends = np.clip(frame_ends, 1, n_samples)
    frame_lengths = np.maximum(frame_ends - frame_starts, 1)

    band_power_matrix = np.zeros((ns5_channels.size, n_frames), dtype=np.float64)
    for row_i in range(ns5_channels.size):
        x_uV = ns5_seg[row_i, i0:i1].astype(np.float64) * ns5_uV_per_bit[row_i]
        x_band = bandpass_filter_matrix(x_uV, ns5_fs, low_hz, high_hz, order=filter_order)
        x_sq = np.square(x_band, dtype=np.float64)
        x_csum = np.concatenate(([0.0], np.cumsum(x_sq)))
        energy = x_csum[frame_ends] - x_csum[frame_starts]
        band_power_matrix[row_i] = np.sqrt(energy / frame_lengths)

    grids = np.zeros((n_frames, 10, 10), dtype=np.float64)
    for frame_i in range(n_frames):
        vals = {int(ch): float(band_power_matrix[row_i, frame_i]) for row_i, ch in enumerate(ns5_channels)}
        grids[frame_i] = np.nan_to_num(channels_to_utah_grid(vals), nan=0.0)
    return grids, edges


def save_utah_grid_video_mp4(
    video_grids: np.ndarray,
    edges_s: np.ndarray,
    video_path: Path,
    *,
    fps: float,
    title: str,
    cbar_label: str,
    ask_before_overwrite: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bool:
    from matplotlib import animation
    import matplotlib.pyplot as plt

    path = Path(video_path)
    if not should_write_output(path, ask_before_overwrite=ask_before_overwrite):
        return False
    if not animation.writers.is_available("ffmpeg"):
        print("FFmpeg writer is not available; install ffmpeg to export MP4 files.")
        return False

    if vmin is None or vmax is None:
        vals = video_grids[np.isfinite(video_grids) & (video_grids > 0)]
        if vals.size > 0:
            auto_vmin = max(0.0, float(np.percentile(vals, 1.0)))
            auto_vmax = float(np.percentile(vals, 99.5))
            if auto_vmax <= auto_vmin:
                auto_vmin = float(np.min(vals))
                auto_vmax = float(np.max(vals))
        else:
            auto_vmin, auto_vmax = 0.0, 1.0
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(video_grids[0], cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ts_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.25"),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    def _update(frame_i: int):
        im.set_data(video_grids[frame_i])
        t0 = edges_s[frame_i]
        t1 = edges_s[frame_i + 1]
        ts_text.set_text(f"{t0:.3f}-{t1:.3f} s")
        return im, ts_text

    _update(0)
    anim = animation.FuncAnimation(fig, _update, frames=video_grids.shape[0], interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(str(path), writer=writer, dpi=180)
    plt.close(fig)
    print("Saved video:", path)
    return True


def plot_spike_rate_utah_maps(
    spike_t_s: np.ndarray,
    spike_ch: np.ndarray,
    *,
    start_s: float,
    interval_s: float,
    total_s: float,
    recording_duration_s: float,
    vmin: float,
    vmax: float,
):
    import matplotlib.pyplot as plt

    end_total_s = min(recording_duration_s, start_s + total_s)
    if end_total_s <= start_s:
        raise ValueError("Invalid spike-rate interval range.")
    interval_starts = np.arange(start_s, end_total_s, interval_s)
    n_intervals = interval_starts.size
    ncols = min(5, n_intervals)
    nrows = int(np.ceil(n_intervals / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, t0 in enumerate(interval_starts):
        ax = axes[idx]
        t1 = min(t0 + interval_s, end_total_s)
        mask = (spike_t_s >= t0) & (spike_t_s < t1)
        ch_win = spike_ch[mask]
        unique_ch, counts = np.unique(ch_win, return_counts=True)
        duration = max(t1 - t0, 1e-9)
        rate_hz = counts / duration
        by_channel = {int(ch): float(rate) for ch, rate in zip(unique_ch, rate_hz)}
        grid = np.nan_to_num(channels_to_utah_grid(by_channel), nan=0.0)
        im = ax.imshow(grid, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{t0:.1f}-{t1:.1f} s")
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_intervals, axes.size):
        axes[idx].axis("off")

    fig.suptitle(f"NEV spike-rate Utah maps ({interval_s:.1f}s bins)", y=1.02)
    fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.35, right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spike rate (Hz)")
    return fig


def build_firing_rate_matrix(
    spike_t_s: np.ndarray,
    spike_ch: np.ndarray,
    *,
    start_s: float,
    total_s: float,
    bin_s: float,
    recording_duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    end_s = min(recording_duration_s, start_s + total_s)
    if end_s <= start_s:
        raise ValueError("Invalid firing-rate range.")
    bin_edges = np.arange(start_s, end_s + bin_s, bin_s)
    if bin_edges[-1] < end_s:
        bin_edges = np.append(bin_edges, end_s)

    present_channels = np.array(sorted(set(np.asarray(spike_ch, dtype=int).tolist())), dtype=int)
    electrode_ids = np.array([int(channel_to_electrode_label(int(ch))) for ch in present_channels], dtype=int)
    order = np.argsort(electrode_ids)
    present_channels = present_channels[order]
    electrode_ids = electrode_ids[order]

    rate_matrix = np.zeros((present_channels.size, bin_edges.size - 1), dtype=float)
    for i, ch in enumerate(present_channels):
        ch_spikes = spike_t_s[spike_ch == ch]
        counts, _ = np.histogram(ch_spikes, bins=bin_edges)
        rate_matrix[i] = counts / np.diff(bin_edges)

    return rate_matrix, bin_edges, present_channels, electrode_ids


def plot_binary_spike_psth_raster(
    spike_t_s: np.ndarray,
    spike_ch: np.ndarray,
    *,
    start_s: float,
    total_s: float,
    bin_s: float,
    recording_duration_s: float,
    channel_ids: Sequence[int] | None = None,
):
    """
    Plot a binary (0/1) spike PSTH and raster for a selected time range.
    Raster rows are channels and columns are time bins.
    """
    import matplotlib.pyplot as plt

    if bin_s <= 0:
        raise ValueError("bin_s must be > 0.")

    if channel_ids is None:
        channels = np.array(sorted(CHANNEL_TO_ELECTRODE.keys()), dtype=int)
    else:
        channels = np.asarray(channel_ids, dtype=int)
    channels = np.unique(channels)
    if channels.size == 0:
        raise ValueError("channel_ids must not be empty.")

    if np.isfinite(recording_duration_s):
        end_limit_s = float(recording_duration_s)
    elif spike_t_s.size:
        end_limit_s = float(np.max(spike_t_s))
    else:
        end_limit_s = start_s + total_s

    end_s = min(end_limit_s, start_s + total_s)
    if end_s <= start_s:
        raise ValueError("Invalid PSTH/raster range.")

    n_bins = int(np.ceil((end_s - start_s) / bin_s))
    n_bins = max(1, n_bins)
    bin_edges = start_s + np.arange(n_bins + 1, dtype=float) * bin_s
    bin_edges[-1] = end_s

    raster_01 = np.zeros((channels.size, n_bins), dtype=np.uint8)

    if spike_t_s.size and spike_ch.size:
        win_mask = (spike_t_s >= start_s) & (spike_t_s < end_s)
        t_win = spike_t_s[win_mask]
        ch_win = np.asarray(spike_ch[win_mask], dtype=int)

        if t_win.size:
            bin_idx = np.searchsorted(bin_edges, t_win, side="right") - 1
            valid_bin = (bin_idx >= 0) & (bin_idx < n_bins)

            row_idx = np.searchsorted(channels, ch_win)
            valid_row = row_idx < channels.size
            valid_ch = np.zeros_like(valid_row, dtype=bool)
            valid_ch[valid_row] = channels[row_idx[valid_row]] == ch_win[valid_row]

            valid = valid_bin & valid_ch
            raster_01[row_idx[valid], bin_idx[valid]] = 1

    psth_active_channels = raster_01.sum(axis=0).astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax_psth, ax_raster) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 3.2]},
    )

    ax_psth.step(bin_centers, psth_active_channels, where="mid", color="black", linewidth=1.0)
    ax_psth.fill_between(bin_centers, 0.0, psth_active_channels, step="mid", color="black", alpha=0.15)
    ax_psth.set_ylabel("Active ch/bin")
    ax_psth.set_title(f"Binary PSTH (0/1 spikes, {bin_s * 1000:.0f} ms bins)")
    ax_psth.set_xlim(start_s, end_s)
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)

    im = ax_raster.imshow(
        raster_01,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=[bin_edges[0], bin_edges[-1], channels[0] - 0.5, channels[-1] + 0.5],
    )
    ax_raster.set_title("Spike raster (all channels, 1=spike present in bin)")
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Channel")
    ax_raster.set_yticks(np.arange(int(channels.min()), int(channels.max()) + 1, 8))
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_raster, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_label("Spike (0/1)")

    fig.tight_layout()
    return fig, raster_01, bin_edges, channels


def plot_firing_rate_heatmap(rate_matrix: np.ndarray, bin_edges: np.ndarray, electrode_ids: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(
        rate_matrix,
        aspect="auto",
        origin="lower",
        extent=[bin_edges[0], bin_edges[-1], electrode_ids[0] - 0.5, electrode_ids[-1] + 0.5],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title("Firing rates over time (one electrode per row)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode ID")
    ax.set_yticks(np.arange(1, 97, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rate (Hz)")
    fig.tight_layout()
    return fig


def plot_firing_rate_traces(
    rate_matrix: np.ndarray,
    bin_edges: np.ndarray,
    electrode_ids: np.ndarray,
    *,
    mode: str = "normalized",
    vertical_span: float = 0.9,
):
    import matplotlib.pyplot as plt

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    global_scale_hz = float(np.percentile(rate_matrix, 95)) if rate_matrix.size else 1.0
    if global_scale_hz <= 0:
        global_scale_hz = 1.0

    fig, ax = plt.subplots(figsize=(14, 11))
    for i in range(rate_matrix.shape[0]):
        row_rate = rate_matrix[i]
        if mode == "normalized":
            row_scale_hz = float(np.percentile(row_rate, 95))
            if row_scale_hz <= 0:
                row_scale_hz = 1.0
            y = electrode_ids[i] + vertical_span * np.clip(row_rate / row_scale_hz, 0.0, 1.25)
        else:
            y = electrode_ids[i] + vertical_span * np.clip(row_rate / global_scale_hz, 0.0, 1.25)
        ax.hlines(electrode_ids[i], bin_edges[0], bin_edges[-1], color="#cfcfcf", lw=0.4, zorder=0)
        ax.plot(bin_centers, y, color="black", lw=0.7, alpha=0.75)

    if mode == "normalized":
        ax.set_title("Firing rate traces over time (normalized per electrode)")
    else:
        ax.set_title(f"Firing rate traces over time (global scale {global_scale_hz:.1f} Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode ID (+ scaled rate)")
    ax.set_ylim(electrode_ids.min() - 1, electrode_ids.max() + 2)
    ax.set_yticks(np.arange(1, 97, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
