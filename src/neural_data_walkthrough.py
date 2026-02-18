#%% Neural Data Walkthrough
#   Antonio Lozano a.lozano@umh.es NBIO, UMH, Spain
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from brpylib import NevFile, NsxFile

try:
    from utils import (
        bandpass_filter_matrix,
        build_band_power_video_grids,
        build_firing_rate_matrix,
        build_spike_rate_video_grids,
        channel_to_electrode_label,
        create_results_dirs,
        load_app_config,
        nev_waveform_bits_to_uV,
        nsx_extended_headers_by_channel,
        nsx_uV_per_bit_from_header,
        pca_cluster_mean_waveforms_by_channel,
        plot_bandpass_traces,
        plot_continuous_traces_ns2_ns5,
        plot_filtered_band_power,
        plot_firing_rate_heatmap,
        plot_firing_rate_traces,
        plot_fourier_power_ns2_ns5,
        plot_nev_waveforms,
        plot_binary_spike_psth_raster,
        plot_spike_rate_utah_maps,
        plot_utah_cluster_mean_waveforms,
        plot_utah_layout,
        resolve_bandpass_range_hz,
        save_figure,
        save_utah_grid_video_mp4,
        slugify_name,
        utah_channel_grid,
        utah_electrode_grid,
    )
except ImportError as exc:
    # Notebook sessions may cache an old `utils` module. Reload once and retry.
    if "plot_utah_layout" not in str(exc):
        raise
    import importlib
    import utils as _utils

    _utils = importlib.reload(_utils)
    bandpass_filter_matrix = _utils.bandpass_filter_matrix
    build_band_power_video_grids = _utils.build_band_power_video_grids
    build_firing_rate_matrix = _utils.build_firing_rate_matrix
    build_spike_rate_video_grids = _utils.build_spike_rate_video_grids
    channel_to_electrode_label = _utils.channel_to_electrode_label
    create_results_dirs = _utils.create_results_dirs
    load_app_config = _utils.load_app_config
    nev_waveform_bits_to_uV = _utils.nev_waveform_bits_to_uV
    nsx_extended_headers_by_channel = _utils.nsx_extended_headers_by_channel
    nsx_uV_per_bit_from_header = _utils.nsx_uV_per_bit_from_header
    pca_cluster_mean_waveforms_by_channel = _utils.pca_cluster_mean_waveforms_by_channel
    plot_bandpass_traces = _utils.plot_bandpass_traces
    plot_continuous_traces_ns2_ns5 = _utils.plot_continuous_traces_ns2_ns5
    plot_filtered_band_power = _utils.plot_filtered_band_power
    plot_firing_rate_heatmap = _utils.plot_firing_rate_heatmap
    plot_firing_rate_traces = _utils.plot_firing_rate_traces
    plot_fourier_power_ns2_ns5 = _utils.plot_fourier_power_ns2_ns5
    plot_nev_waveforms = _utils.plot_nev_waveforms
    plot_binary_spike_psth_raster = _utils.plot_binary_spike_psth_raster
    plot_spike_rate_utah_maps = _utils.plot_spike_rate_utah_maps
    plot_utah_cluster_mean_waveforms = _utils.plot_utah_cluster_mean_waveforms
    plot_utah_layout = _utils.plot_utah_layout
    resolve_bandpass_range_hz = _utils.resolve_bandpass_range_hz
    save_figure = _utils.save_figure
    save_utah_grid_video_mp4 = _utils.save_utah_grid_video_mp4
    slugify_name = _utils.slugify_name
    utah_channel_grid = _utils.utah_channel_grid
    utah_electrode_grid = _utils.utah_electrode_grid


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.0,
        "lines.linewidth": 0.9,
        "figure.dpi": 130,
    }
)

# %%
# User settings.
CONFIG = load_app_config()
DATA_DIR = CONFIG["data_dir"]
RECORDING_DIR_NAME = CONFIG["recording_dir_name"]
RECORDING_NAME = CONFIG["recording_name"]
WINDOW_S = CONFIG["window_s"]

READ_NEV_WAVEFORMS = True
SAVE_OUTPUTS = True
ASK_BEFORE_OVERWRITE = False  # Set True to ask before overwriting figures/videos.

WAVEFORM_CHANNEL = 16
MAX_WAVEFORMS_TO_PLOT = 500
WAVEFORM_VIEW_MODE = "cluster"  # "std", "all", or "cluster"
WAVEFORM_SITE_NORMALIZATION = "per_site"  # "per_site" or "global"

CLUSTER_MAX_UNITS_PER_CHANNEL = 3
CLUSTER_MAX_SPIKES_PER_CHANNEL = 2000
CLUSTER_MIN_SPIKES_FOR_CLUSTERING = 80
CLUSTER_MIN_CLUSTER_SIZE = 20
CLUSTER_N_JOBS = 0  # 0/None = auto

CHANNELS_TO_PLOT = [2, 16, 48, 96]
WINDOW_VISUALIZE_START_S = 0.0
WINDOW_VISUALIZE_END_S = 1.0  # None means until end

FILTER_BAND = "alpha"  # "alpha", "beta", "gamma", "high_gamma"
FILTER_START_S = WINDOW_VISUALIZE_START_S
FILTER_END_S = WINDOW_VISUALIZE_END_S
FILTER_ORDER = 4

SPIKE_RATE_START_S = 0.0
SPIKE_RATE_INTERVAL_S = 1.0
SPIKE_RATE_TOTAL_S = 10.0
SPIKE_RATE_VMIN = 0.0
SPIKE_RATE_VMAX = 200.0

RATE_VIDEO_BIN_MS = 50  # Common values: 10 or 40
RATE_VIDEO_START_S = SPIKE_RATE_START_S
RATE_VIDEO_TOTAL_S = SPIKE_RATE_TOTAL_S
RATE_VIDEO_FPS = 25
RATE_VIDEO_VMIN = SPIKE_RATE_VMIN
RATE_VIDEO_VMAX = SPIKE_RATE_VMAX

BAND_VIDEO_BAND = FILTER_BAND
BAND_VIDEO_BIN_MS = RATE_VIDEO_BIN_MS
BAND_VIDEO_START_S = RATE_VIDEO_START_S
BAND_VIDEO_TOTAL_S = RATE_VIDEO_TOTAL_S
BAND_VIDEO_FPS = RATE_VIDEO_FPS
BAND_VIDEO_FILTER_ORDER = FILTER_ORDER

RATE_TRACE_START_S = 0.0
RATE_TRACE_TOTAL_S = 10.0
RATE_TRACE_BIN_S = 0.2
RATE_TRACE_MODE = "normalized"  # "normalized" or "absolute"
RATE_TRACE_VERTICAL_SPAN = 0.9
PSTH_START_S = 0.0
PSTH_TOTAL_S = 10.0
PSTH_BIN_S = 0.02  # 20 ms bins
PSTH_CHANNEL_IDS = np.arange(1, 97, dtype=int)  # All channels in raster

CLOSE_FILES_AT_END = False  # Keep False for interactive exploration.

OUTPUT_DIRS = create_results_dirs(CONFIG["project_root"], RECORDING_DIR_NAME, RECORDING_NAME)
PLOTS_DIR = OUTPUT_DIRS["plots_dir"]
VIDEOS_DIR = OUTPUT_DIRS["videos_dir"]

nev_path = DATA_DIR / f"{RECORDING_NAME}.nev"
ns2_path = DATA_DIR / f"{RECORDING_NAME}.ns2"
ns5_path = DATA_DIR / f"{RECORDING_NAME}.ns5"
for p in [nev_path, ns2_path, ns5_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

print("Data directory:", DATA_DIR)
print("Recording dir:", RECORDING_DIR_NAME)
print("Recording name:", RECORDING_NAME)
print("Window (s):", WINDOW_S, "(None = full recording)")
print("Plots output dir:", PLOTS_DIR)
print("Videos output dir:", VIDEOS_DIR)


def _save(fig, filename: str) -> None:
    if not SAVE_OUTPUTS:
        return
    save_figure(fig, PLOTS_DIR, filename, ask_before_overwrite=ASK_BEFORE_OVERWRITE, dpi=220)


# %%
# Utah layout sanity-check.
LAYOUT_CHANNELS = utah_channel_grid()
LAYOUT_ELECTRODES = utah_electrode_grid()

fig = plot_utah_layout(LAYOUT_CHANNELS, LAYOUT_ELECTRODES)
_save(fig, "01_utah_layout_labels.png")
plt.show()

# %%
# Load raw NEV data (kept exposed for users).
NEV_FILE = NevFile(str(nev_path))
NEV_DATA = NEV_FILE.getdata(wave_read="read" if READ_NEV_WAVEFORMS else "no_read")
SPIKE_EVENTS = NEV_DATA.get("spike_events", {})
SPIKE_TS = np.asarray(SPIKE_EVENTS.get("TimeStamps", []), dtype=np.int64)
SPIKE_CH = np.asarray(SPIKE_EVENTS.get("Channel", []), dtype=np.int64)
SPIKE_UNIT = np.asarray(SPIKE_EVENTS.get("Unit", []), dtype=np.int64)

TIMESTAMP_HZ = float(NEV_FILE.basic_header["TimeStampResolution"])
RECORDING_DURATION_S = float(SPIKE_TS.max() / TIMESTAMP_HZ) if SPIKE_TS.size else np.nan
SPIKE_T_S = SPIKE_TS / TIMESTAMP_HZ if SPIKE_TS.size else np.array([], dtype=float)

print("NEV data keys:", list(NEV_DATA.keys()))
print("NEV timestamp resolution (Hz):", TIMESTAMP_HZ)
print("Total spikes:", SPIKE_TS.size)
print("Unique channels in spikes:", np.unique(SPIKE_CH).size if SPIKE_CH.size else 0)
print("Unique units:", np.unique(SPIKE_UNIT) if SPIKE_UNIT.size else [])
print("Approx duration from NEV timestamps (s):", RECORDING_DURATION_S)

# %%
# Spike summary.
spike_unique_ch, spike_counts = np.unique(SPIKE_CH, return_counts=True)
spike_rate_hz = spike_counts / RECORDING_DURATION_S if spike_counts.size else np.array([])
print("First 12 spike counts by channel:", list(zip(spike_unique_ch[:12], spike_counts[:12])))
print("First 12 spike rates (Hz):", list(zip(spike_unique_ch[:12], np.round(spike_rate_hz[:12], 3))))
if spike_rate_hz.size:
    print(
        "Spike-rate summary (Hz): min/median/max =",
        np.round(spike_rate_hz.min(), 3),
        np.round(np.median(spike_rate_hz), 3),
        np.round(spike_rate_hz.max(), 3),
    )

#%% Plot histogram of spike rates across channels.

plt.figure(figsize=(8, 4), dpi = 200)
plt.hist(spike_rate_hz, bins = 200, color = 'k')
plt.title("Histogram of spike rates (Hz) across channels")
plt.xlabel("Spike rate (Hz)")
plt.axis('tight')
plt.show()

# %% Show waveforms for a single channel
# Optional NEV waveform detail for one channel.
WAVEFORM_CHANNEL = 95
NEV_WAVEFORMS_UV = None
if READ_NEV_WAVEFORMS and "Waveforms" in SPIKE_EVENTS:
    NEV_WAVEFORMS_UV = nev_waveform_bits_to_uV(np.asarray(SPIKE_EVENTS["Waveforms"], dtype=np.int16))
    wf = NEV_WAVEFORMS_UV[SPIKE_CH == WAVEFORM_CHANNEL]
    if wf.shape[0] > MAX_WAVEFORMS_TO_PLOT:
        wf = wf[:MAX_WAVEFORMS_TO_PLOT]
    print(f"Waveforms for channel {WAVEFORM_CHANNEL}:", wf.shape, "(uV)")
    if wf.size:
        fig = plot_nev_waveforms(wf, WAVEFORM_CHANNEL, view_mode=WAVEFORM_VIEW_MODE)
        _save(fig, f"02_nev_waveforms_ch_{WAVEFORM_CHANNEL}_{WAVEFORM_VIEW_MODE}.png")
        plt.show()

# %%
# Utah overlay of PCA-cluster mean waveforms.

CLUSTER_N_JOBS = -1  # 0/None means auto-detect number of CPU cores for parallel processing. 

MEAN_WAVEFORMS_BY_CHANNEL: dict[int, list[np.ndarray]] = {}
if NEV_WAVEFORMS_UV is not None:
    t0 = time.perf_counter()
    MEAN_WAVEFORMS_BY_CHANNEL = pca_cluster_mean_waveforms_by_channel(
        NEV_WAVEFORMS_UV,
        SPIKE_CH,
        max_clusters=CLUSTER_MAX_UNITS_PER_CHANNEL,
        pca_components=3,
        max_spikes_per_channel=CLUSTER_MAX_SPIKES_PER_CHANNEL,
        min_spikes_for_clustering=CLUSTER_MIN_SPIKES_FOR_CLUSTERING,
        min_cluster_size=CLUSTER_MIN_CLUSTER_SIZE,
        min_cluster_fraction=0.05,
        random_state=0,
        n_jobs=CLUSTER_N_JOBS,
    )
    dt_s = time.perf_counter() - t0
    print("PCA+cluster mean waveforms computed for", len(MEAN_WAVEFORMS_BY_CHANNEL), "channels in", f"{dt_s:.2f}s")
    fig = plot_utah_cluster_mean_waveforms(
        LAYOUT_CHANNELS,
        LAYOUT_ELECTRODES,
        MEAN_WAVEFORMS_BY_CHANNEL,
        max_units_per_channel=CLUSTER_MAX_UNITS_PER_CHANNEL,
        normalization=WAVEFORM_SITE_NORMALIZATION,
    )
    _save(fig, "03_nev_cluster_mean_waveforms_utah.png")
    plt.show()

# %%
# Binary spike PSTH + all-channel raster for selected times.
fig, BINARY_RASTER_01, BINARY_BIN_EDGES, BINARY_RASTER_CHANNELS = plot_binary_spike_psth_raster(
    SPIKE_T_S,
    SPIKE_CH,
    start_s=PSTH_START_S,
    total_s=PSTH_TOTAL_S,
    bin_s=PSTH_BIN_S,
    recording_duration_s=RECORDING_DURATION_S,
    channel_ids=PSTH_CHANNEL_IDS,
)
_save(fig, "03b_binary_spike_psth_raster_all_channels.png")
plt.show()

# %%
# Load raw NS2/NS5 data (kept exposed for users).
data_time_s = "all" if WINDOW_S is None else WINDOW_S
NS2_FILE = NsxFile(str(ns2_path))
NS2_DATA = NS2_FILE.getdata(data_time_s=data_time_s)
NS5_FILE = NsxFile(str(ns5_path))
NS5_DATA = NS5_FILE.getdata(data_time_s=data_time_s)

NS2_SEG = NS2_DATA["data"][0]
NS5_SEG = NS5_DATA["data"][0]
NS2_FS = float(NS2_DATA["samp_per_s"])
NS5_FS = float(NS5_DATA["samp_per_s"])
NS2_CHANNELS = np.asarray(NS2_DATA["elec_ids"], dtype=int)
NS5_CHANNELS = np.asarray(NS5_DATA["elec_ids"], dtype=int)

NS2_HEADER_BY_CH = nsx_extended_headers_by_channel(NS2_FILE.extended_headers)
NS5_HEADER_BY_CH = nsx_extended_headers_by_channel(NS5_FILE.extended_headers)
NS2_UV_PER_BIT = np.array([nsx_uV_per_bit_from_header(NS2_HEADER_BY_CH[int(ch)]) for ch in NS2_CHANNELS], dtype=float)
NS5_UV_PER_BIT = np.array([nsx_uV_per_bit_from_header(NS5_HEADER_BY_CH[int(ch)]) for ch in NS5_CHANNELS], dtype=float)

print("NS2 shape:", NS2_SEG.shape, "fs:", NS2_FS)
print("NS5 shape:", NS5_SEG.shape, "fs:", NS5_FS)
print("NS2 channels first 12:", NS2_CHANNELS[:12])
print("NS5 channels first 12:", NS5_CHANNELS[:12])

# %%
# Continuous NS2/NS5 traces for selected channels.
fig = plot_continuous_traces_ns2_ns5(
    NS2_SEG,
    NS2_FS,
    NS2_CHANNELS,
    NS2_UV_PER_BIT,
    NS5_SEG,
    NS5_FS,
    NS5_CHANNELS,
    NS5_UV_PER_BIT,
    CHANNELS_TO_PLOT,
    start_s=WINDOW_VISUALIZE_START_S,
    end_s=WINDOW_VISUALIZE_END_S,
    title_label=RECORDING_NAME,
)
_save(fig, "04_continuous_ns2_ns5_traces.png")
plt.show()

# %%
# Fourier power for selected channels (NS2 + NS5).
fig = plot_fourier_power_ns2_ns5(
    NS2_SEG,
    NS2_FS,
    NS2_CHANNELS,
    NS2_UV_PER_BIT,
    NS5_SEG,
    NS5_FS,
    NS5_CHANNELS,
    NS5_UV_PER_BIT,
    CHANNELS_TO_PLOT,
    start_s=WINDOW_VISUALIZE_START_S,
    end_s=WINDOW_VISUALIZE_END_S,
    title_label=RECORDING_NAME,
)
_save(fig, "05_fourier_power_ns2_ns5.png")
plt.show()

# %%
# Bandpass traces and filtered power (NS5).

FILTER_BAND = "beta"  # "alpha", "beta", "gamma", "high_gamma"
low_hz, high_hz = resolve_bandpass_range_hz(FILTER_BAND)
ns5_total_s = NS5_SEG.shape[1] / NS5_FS
filter_start_s = min(max(FILTER_START_S, 0.0), ns5_total_s)
filter_end_s = ns5_total_s if FILTER_END_S is None else min(FILTER_END_S, ns5_total_s)
if filter_end_s <= filter_start_s:
    raise ValueError("FILTER_END_S must be > FILTER_START_S.")
filter_i0 = int(np.floor(filter_start_s * NS5_FS))
filter_i1 = int(np.ceil(filter_end_s * NS5_FS))
filter_t = np.arange(filter_i0, filter_i1) / NS5_FS

RAW_NS5_UV = []
for ch in CHANNELS_TO_PLOT:
    row_ns5 = int(np.where(NS5_CHANNELS == ch)[0][0])
    RAW_NS5_UV.append(NS5_SEG[row_ns5, filter_i0:filter_i1].astype(np.float64) * NS5_UV_PER_BIT[row_ns5])
RAW_NS5_UV = np.asarray(RAW_NS5_UV)
FILTERED_NS5_UV = bandpass_filter_matrix(RAW_NS5_UV, NS5_FS, low_hz, high_hz, order=FILTER_ORDER)

fig = plot_bandpass_traces(
    FILTERED_NS5_UV,
    filter_t,
    CHANNELS_TO_PLOT,
    band_label=f"{FILTER_BAND} ({low_hz:.0f}-{high_hz:.0f} Hz), order {FILTER_ORDER}",
)
_save(fig, f"06_ns5_bandpass_{FILTER_BAND}_traces.png")
plt.show()

fig = plot_filtered_band_power(
    RAW_NS5_UV,
    FILTERED_NS5_UV,
    NS5_FS,
    CHANNELS_TO_PLOT,
    low_hz=low_hz,
    high_hz=high_hz,
    band_label=FILTER_BAND,
)
_save(fig, f"07_ns5_bandpass_{FILTER_BAND}_power.png")
plt.show()

# %%
# Spike-rate Utah maps.
fig = plot_spike_rate_utah_maps(
    SPIKE_T_S,
    SPIKE_CH,
    start_s=SPIKE_RATE_START_S,
    interval_s=SPIKE_RATE_INTERVAL_S,
    total_s=SPIKE_RATE_TOTAL_S,
    recording_duration_s=RECORDING_DURATION_S,
    vmin=SPIKE_RATE_VMIN,
    vmax=SPIKE_RATE_VMAX,
)
_save(fig, f"08_utah_spike_rate_maps_{SPIKE_RATE_INTERVAL_S:.3f}s.png")
plt.show()

# %%
# Spike-rate Utah video.
if SAVE_OUTPUTS:
    rate_video_grids, rate_video_edges = build_spike_rate_video_grids(
        SPIKE_T_S,
        SPIKE_CH,
        start_s=RATE_VIDEO_START_S,
        total_s=RATE_VIDEO_TOTAL_S,
        bin_ms=RATE_VIDEO_BIN_MS,
        end_limit_s=RECORDING_DURATION_S,
    )
    rate_video_filename = (
        f"{slugify_name(RECORDING_DIR_NAME)}__{RECORDING_NAME}__utah_spike_rate_{int(RATE_VIDEO_BIN_MS)}ms.mp4"
    )
    save_utah_grid_video_mp4(
        rate_video_grids,
        rate_video_edges,
        VIDEOS_DIR / rate_video_filename,
        fps=RATE_VIDEO_FPS,
        title="NEV spike-rate Utah map over time",
        cbar_label="Spike rate (Hz)",
        ask_before_overwrite=ASK_BEFORE_OVERWRITE,
        vmin=RATE_VIDEO_VMIN,
        vmax=RATE_VIDEO_VMAX,
    )

# %%
# NS5 band-power Utah video.

BAND_VIDEO_BAND = FILTER_BAND

if SAVE_OUTPUTS:
    band_video_grids, band_video_edges = build_band_power_video_grids(
        NS5_SEG,
        NS5_UV_PER_BIT,
        NS5_CHANNELS,
        NS5_FS,
        start_s=BAND_VIDEO_START_S,
        total_s=BAND_VIDEO_TOTAL_S,
        bin_ms=BAND_VIDEO_BIN_MS,
        band_name=BAND_VIDEO_BAND,
        filter_order=BAND_VIDEO_FILTER_ORDER,
    )
    band_video_filename = (
        f"{slugify_name(RECORDING_DIR_NAME)}__{RECORDING_NAME}"
        f"__utah_bandpower_{slugify_name(BAND_VIDEO_BAND)}_{int(BAND_VIDEO_BIN_MS)}ms.mp4"
    )
    save_utah_grid_video_mp4(
        band_video_grids,
        band_video_edges,
        VIDEOS_DIR / band_video_filename,
        fps=BAND_VIDEO_FPS,
        title=f"NS5 {BAND_VIDEO_BAND} band Utah power over time",
        cbar_label="Band power (uV RMS)",
        ask_before_overwrite=ASK_BEFORE_OVERWRITE,
        vmin=None,
        vmax=None,
    )

# %%
# Firing-rate heatmap + traces.
RATE_MATRIX, BIN_EDGES, PRESENT_CHANNELS, ELECTRODE_IDS = build_firing_rate_matrix(
    SPIKE_T_S,
    SPIKE_CH,
    start_s=RATE_TRACE_START_S,
    total_s=RATE_TRACE_TOTAL_S,
    bin_s=RATE_TRACE_BIN_S,
    recording_duration_s=RECORDING_DURATION_S,
)

fig = plot_firing_rate_heatmap(RATE_MATRIX, BIN_EDGES, ELECTRODE_IDS)
_save(fig, "09_firing_rate_heatmap_over_time.png")
plt.show()

fig = plot_firing_rate_traces(
    RATE_MATRIX,
    BIN_EDGES,
    ELECTRODE_IDS,
    mode=RATE_TRACE_MODE,
    vertical_span=RATE_TRACE_VERTICAL_SPAN,
)
_save(fig, "10_firing_rate_traces_over_time.png")
plt.show()

# %%
# Optional cleanup cell.
if CLOSE_FILES_AT_END:
    NEV_FILE.close()
    NS2_FILE.close()
    NS5_FILE.close()
    print("Closed NEV/NS2/NS5 file handles.")
else:
    print("Raw data handles remain open: NEV_FILE, NS2_FILE, NS5_FILE.")
