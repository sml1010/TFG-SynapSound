#%% Antonio Lozano a.lozano@umh.es NBIO, UMH, Spain

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from brpylib import NevFile, NsxFile

from utils import (
    channel_to_electrode_label,
    channels_to_utah_grid,
    load_app_config,
    nsx_extended_headers_by_channel,
    nsx_uV_per_bit_from_header,
)



def print_section(title: str) -> None:
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def summarize_dict_of_arrays(obj: dict[str, Any], prefix: str = "") -> None:
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            print(f"{name}: dict keys={list(value.keys())}")
            summarize_dict_of_arrays(value, prefix=f"{name}.")
            continue
        if isinstance(value, list):
            if value and isinstance(value[0], (np.generic, int, float)):
                arr = np.asarray(value)
                print(
                    f"{name}: list len={len(value)} dtype={arr.dtype} "
                    f"min={arr.min()} max={arr.max()}"
                )
            else:
                print(f"{name}: list len={len(value)} first_type={type(value[0]).__name__ if value else None}")
            continue
        if isinstance(value, np.ndarray):
            print(
                f"{name}: ndarray shape={value.shape} dtype={value.dtype} "
                f"min={value.min()} max={value.max()}"
            )
            continue
        print(f"{name}: {type(value).__name__} value={value}")


def main() -> None:
    config = load_app_config()

    parser = argparse.ArgumentParser(description="Debug/explore BRPylib content for one recording.")
    parser.add_argument(
        "--recording-name",
        "--stem",
        dest="recording_name",
        default=config["recording_name"],
        help="Recording name, e.g. spontaneous_initial0018",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=config["window_s"],
        help="Time window for NSx inspection (omit to load full recording).",
    )
    parser.add_argument("--read-waveforms", action="store_true", help="Read NEV waveforms for shape/stat checks")
    args = parser.parse_args()

    data_dir = config["data_dir"]
    recording_name = args.recording_name
    paths = {
        "nev": data_dir / f"{recording_name}.nev",
        "ns2": data_dir / f"{recording_name}.ns2",
        "ns5": data_dir / f"{recording_name}.ns5",
    }

    print_section("INPUT")
    print(f"project_root={config['project_root']}")
    print(f"data_dir={data_dir}")
    print(f"recording_name={recording_name}")
    print(f"window_s={args.window_s} (None means full recording)")
    for ext, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        print(f"{ext}: {path.name} size_mb={path.stat().st_size / (1024 ** 2):.2f}")

    print_section("NEV HEADER")
    nev_file = NevFile(str(paths["nev"]))
    print(f"basic_header keys={list(nev_file.basic_header.keys())}")
    for key in ["FileTypeID", "FileSpec", "TimeStampResolution", "SampleTimeResolution", "BytesInDataPackets"]:
        if key in nev_file.basic_header:
            print(f"basic_header[{key}]={nev_file.basic_header[key]}")
    print(f"extended_headers count={len(nev_file.extended_headers)}")
    if nev_file.extended_headers:
        packet_ids = [h.get("PacketID", None) for h in nev_file.extended_headers[:8]]
        print(f"first_extended_packet_ids={packet_ids}")

    print_section("NEV DATA")
    nev_data = nev_file.getdata(wave_read="read" if args.read_waveforms else "no_read")
    summarize_dict_of_arrays(nev_data)
    spike_events = nev_data.get("spike_events", {})
    spike_channels = np.asarray(spike_events.get("Channel", []), dtype=int)
    spike_units = np.asarray(spike_events.get("Unit", []), dtype=int)
    spike_timestamps = np.asarray(spike_events.get("TimeStamps", []), dtype=np.int64)
    print(f"spike_count={spike_channels.size}")
    if spike_channels.size > 0:
        unique_ch, ch_counts = np.unique(spike_channels, return_counts=True)
        print(f"unique_spike_channels={unique_ch.size} min={unique_ch.min()} max={unique_ch.max()}")
        print(f"spike_counts_first_10={list(zip(unique_ch[:10], ch_counts[:10]))}")
        unique_units, unit_counts = np.unique(spike_units, return_counts=True)
        print(f"unit_distribution={dict(zip(unique_units.tolist(), unit_counts.tolist()))}")
        print(f"timestamp_range=({spike_timestamps.min()}, {spike_timestamps.max()})")

        by_channel = {int(ch): int(count) for ch, count in zip(unique_ch, ch_counts)}
        grid = channels_to_utah_grid(by_channel)
        print(f"utah_grid_valid_values={np.isfinite(grid).sum()} shape={grid.shape}")
        print("utah_grid_spike_counts:")
        print(np.array2string(grid, precision=1, suppress_small=False, max_line_width=220))
        mapped_example = [(int(ch), channel_to_electrode_label(int(ch))) for ch in unique_ch[:12]]
        print(f"channel_to_electrode_example={mapped_example}")
    nev_file.close()

    nsx_outputs: dict[str, dict[str, Any]] = {}
    for ext in ("ns2", "ns5"):
        print_section(f"{ext.upper()} HEADER")
        nsx_file = NsxFile(str(paths[ext]))
        print(f"basic_header keys={list(nsx_file.basic_header.keys())}")
        for key in ["FileTypeID", "FileSpec", "ChannelCount", "Period", "SampleResolution", "TimeStampResolution"]:
            if key in nsx_file.basic_header:
                print(f"basic_header[{key}]={nsx_file.basic_header[key]}")
        print(f"extended_headers count={len(nsx_file.extended_headers)}")
        if nsx_file.extended_headers:
            print(f"extended_header_keys={list(nsx_file.extended_headers[0].keys())}")
            headers_by_ch = nsx_extended_headers_by_channel(nsx_file.extended_headers)
            preview_channels = [1, 2, 16, 32, 64, 96]
            print("extended_header_preview: channel electrode units uV_per_bit high_corner low_corner")
            for ch in preview_channels:
                if ch not in headers_by_ch:
                    continue
                h = headers_by_ch[ch]
                units = str(h.get("Units", "")).strip("\x00")
                uV_per_bit = nsx_uV_per_bit_from_header(h)
                print(
                    ch,
                    channel_to_electrode_label(ch),
                    units,
                    round(float(uV_per_bit), 6),
                    h.get("HighFreqCorner"),
                    h.get("LowFreqCorner"),
                )

        print_section(f"{ext.upper()} DATA")
        data_time_s = "all" if args.window_s is None else args.window_s
        data = nsx_file.getdata(data_time_s=data_time_s, full_timestamps=False)
        nsx_outputs[ext] = data
        print(f"keys={list(data.keys())}")
        print(f"samp_per_s={data['samp_per_s']}")
        print(f"elec_ids count={len(data['elec_ids'])} first12={data['elec_ids'][:12]}")
        print(f"data_headers count={len(data['data_headers'])} first={data['data_headers'][0]}")
        seg0 = data["data"][0]
        print(f"segment0 shape={seg0.shape} dtype={seg0.dtype}")
        print(f"segment0 min={seg0.min()} max={seg0.max()} mean={seg0.mean():.3f} std={seg0.std():.3f}")
        headers_by_ch = nsx_extended_headers_by_channel(nsx_file.extended_headers)
        gains = np.array([nsx_uV_per_bit_from_header(headers_by_ch[int(ch)]) for ch in data["elec_ids"]], dtype=float)
        seg0_min_uV = np.min(seg0, axis=1) * gains
        seg0_max_uV = np.max(seg0, axis=1) * gains
        seg0_mean_uV = np.mean(seg0, axis=1) * gains
        seg0_std_uV = np.std(seg0, axis=1) * gains
        print(
            f"segment0_uV min={seg0_min_uV.min():.3f} max={seg0_max_uV.max():.3f} "
            f"mean={seg0_mean_uV.mean():.3f} std={seg0_std_uV.mean():.3f}"
        )
        rms = np.sqrt(np.mean(seg0.astype(np.float64) ** 2, axis=1))
        print(f"rms first10={rms[:10]}")
        rms_by_channel = {int(ch): float(val) for ch, val in zip(data["elec_ids"], rms)}
        rms_grid = channels_to_utah_grid(rms_by_channel)
        print(f"rms_grid finite={np.isfinite(rms_grid).sum()} nan={np.isnan(rms_grid).sum()}")
        nsx_file.close()

    print_section("COHERENCE")
    ns2_channels = np.asarray(nsx_outputs["ns2"]["elec_ids"], dtype=int)
    ns5_channels = np.asarray(nsx_outputs["ns5"]["elec_ids"], dtype=int)
    print(f"ns2_unique_channels={ns2_channels.size} ns5_unique_channels={ns5_channels.size}")
    print(f"channel_sets_equal={set(ns2_channels.tolist()) == set(ns5_channels.tolist())}")
    if spike_channels.size > 0:
        print(f"nev_channels_subset_of_nsx={set(np.unique(spike_channels).tolist()).issubset(set(ns2_channels.tolist()))}")

    ns2_seg = nsx_outputs["ns2"]["data"][0]
    ns5_seg = nsx_outputs["ns5"]["data"][0]
    ns2_dur = ns2_seg.shape[1] / float(nsx_outputs["ns2"]["samp_per_s"])
    ns5_dur = ns5_seg.shape[1] / float(nsx_outputs["ns5"]["samp_per_s"])
    print(f"window_duration_ns2={ns2_dur:.6f}s window_duration_ns5={ns5_dur:.6f}s")
    print(f"loaded_duration_difference={abs(ns2_dur - ns5_dur):.6f}s")
    print("DONE")


if __name__ == "__main__":
    main()
