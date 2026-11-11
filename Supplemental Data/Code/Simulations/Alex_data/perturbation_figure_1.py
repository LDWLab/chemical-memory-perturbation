import re
import ast
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams["svg.fonttype"] = "none"

SECONDS_PER_DAY = 24 * 3600


# =============================================================================
# User settings
# =============================================================================

PERTURBED_FILE = "timeseries_values_8mer_pert.txt"
NONPERTURBED_FILE = "timeseries_values_8mer_Nopert.txt"

OLIGOMER_LENGTH = 7
THRESHOLD = 1e-5

TARGETS = np.array([0.01, 0.99, 2.99, 5.99, 9.99], dtype=float)

RAW_PERT_CSV = f"time_evolution_{OLIGOMER_LENGTH}mer_pert.csv"
RAW_NOPERT_CSV = f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert.csv"

NORM_PERT_CSV = f"time_evolution_{OLIGOMER_LENGTH}mer_pert_minmax_normalized_to_pert.csv"
NORM_NOPERT_CSV = f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_minmax_normalized_to_pert.csv"


# =============================================================================
# Helpers
# =============================================================================

def simple_name(label):
    return label.replace("M_", "").replace("P_", "").replace("_", "")


def sort_key(label):
    s = simple_name(label)
    return (len(s), [int(c) for c in s])


def parse_value(text):
    text = text.strip()

    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return float(text)
        except Exception:
            return text


def water_concentration_from_params(t_phys, params):
    water_mode = str(params.get("water_mode", "none")).strip()

    if water_mode == "square":
        wp = params.get("water_params", {})
        c_dry = float(wp.get("c_dry", 1.0))
        c_wet = float(wp.get("c_wet", 1.0))
        period = float(wp.get("period", SECONDS_PER_DAY))
        duty = float(wp.get("duty_cycle", 0.5))

        phase = (t_phys % period) / period
        return c_dry if phase < duty else c_wet

    if water_mode == "constant":
        wp = params.get("water_params", {})
        return float(wp.get("c", 1.0))

    return 1.0


def amide_bond_count_from_simple_name(s):
    if len(s) <= 1:
        return 0
    return s[1:].count("1")


def type1_symbol_count_from_simple_name(s):
    return s.count("1")


# =============================================================================
# Readers
# =============================================================================

def read_new_dict_like_format(lines):
    params = {}
    data = {}

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "[Simulation parameters]":
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    break
                if s.startswith("- ") and ":" in s:
                    key, val = s[2:].split(":", 1)
                    params[key.strip()] = parse_value(val)
                i += 1

        elif line == "t_phys_s:":
            i += 1
            block = ""
            while i < len(lines):
                block += lines[i].strip()
                if "]" in lines[i]:
                    break
                i += 1

            data["time"] = np.array(ast.literal_eval(block), dtype=float).tolist()
            i += 1

        elif line == "y:":
            i += 1
            while i < len(lines):
                seq_line = lines[i].rstrip()
                m = re.match(r"^\s*([0-9]+):\s*$", seq_line)

                if not m:
                    i += 1
                    continue

                seq = m.group(1)
                i += 1
                block = ""

                while i < len(lines):
                    block += lines[i].strip()
                    if "]" in lines[i]:
                        break
                    i += 1

                data[seq] = list(map(float, ast.literal_eval(block)))
                i += 1

            break

        else:
            i += 1

    if "time" not in data:
        raise ValueError("Could not find 't_phys_s:' in the file.")

    n = len(data["time"])
    for k, v in data.items():
        if len(v) != n:
            raise ValueError(f"Column '{k}' has {len(v)} values, but time has {n}.")

    t_phys_arr = np.asarray(data["time"], dtype=float)

    data["W_v"] = [
        water_concentration_from_params(t, params)
        for t in t_phys_arr
    ]

    return data, params


def read_old_csv_format(lines):
    header_idx = None

    for idx, line in enumerate(lines):
        if line.strip() == "":
            header_idx = idx + 1
            break

    if header_idx is None or header_idx >= len(lines):
        raise ValueError("Could not find CSV header.")

    headers = lines[header_idx].strip().split(",")
    data = {h: [] for h in headers}

    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue

        values = line.strip().split(",")

        for h, v in zip(headers, values):
            data[h].append(float(v))

    if "W_v" not in data:
        data["W_v"] = [1.0] * len(data["time"])

    return data, {}


def read_timeseries(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text = "".join(lines)

    if "t_phys_s:" in text and "\ny:" in text:
        return read_new_dict_like_format(lines)

    return read_old_csv_format(lines)


# =============================================================================
# Export raw oligomer CSV
# =============================================================================

def export_raw_oligomer_csv(input_file, output_csv, oligomer_length, targets):
    data, _ = read_timeseries(input_file)

    time_days = np.asarray(data["time"], dtype=float) / SECONDS_PER_DAY

    polymer_species = [
        h for h in data.keys()
        if h not in ("time", "W_v")
    ]

    ordered_species = sorted(polymer_species, key=sort_key)

    target_species = []

    for h in ordered_species:
        s = simple_name(h)

        if len(s) != oligomer_length:
            continue

        if not set(s) <= {"0", "1"}:
            continue

        target_species.append(h)

    idx_targets = np.argmin(
        np.abs(time_days[:, None] - targets[None, :]),
        axis=0,
    )

    sampled_times = time_days[idx_targets]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = (
            ["species", "simple_name", "oligomer_length"]
            + [f"t={t:.5f}_days" for t in sampled_times]
        )

        writer.writerow(header)

        for h in target_species:
            arr = np.asarray(data[h], dtype=float)
            values = arr[idx_targets]
            writer.writerow([h, simple_name(h), oligomer_length] + list(values))

    print(f"Saved raw CSV: {output_csv}")
    print(f"Selected {len(target_species)} length-{oligomer_length} species from {input_file}")


# =============================================================================
# Read oligomer CSV
# =============================================================================

def read_oligomer_csv(filename):
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        rows = []

        for row in reader:
            rows.append({
                "species": row[0],
                "simple_name": row[1],
                "oligomer_length": int(row[2]),
                "values": np.array(row[3:], dtype=float),
            })

    times = []

    for h in header[3:]:
        t = h.replace("t=", "").replace("_days", "")
        times.append(float(t))

    return header, rows, np.array(times, dtype=float)


# =============================================================================
# Normalize to perturbed min-max
# =============================================================================

def normalize_to_perturbed_minmax(input_csv, output_csv, reference_csv):
    _, ref_rows, _ = read_oligomer_csv(reference_csv)

    reference_minmax = {}

    for row in ref_rows:
        values = row["values"]
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        reference_minmax[row["species"]] = {
            "min": vmin,
            "range": vmax - vmin,
        }

    header, rows, _ = read_oligomer_csv(input_csv)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            species = row["species"]
            values = row["values"]

            if species not in reference_minmax:
                print(f"WARNING: {species} not found in perturbed reference file.")
                continue

            vmin = reference_minmax[species]["min"]
            vrange = reference_minmax[species]["range"]

            if vrange == 0:
                normalized = np.zeros_like(values, dtype=float)
            else:
                normalized = (values - vmin) / vrange

            writer.writerow([
                row["species"],
                row["simple_name"],
                row["oligomer_length"],
            ] + list(normalized))

    print(f"Saved normalized CSV: {output_csv}")


# =============================================================================
# Plotting
# =============================================================================

PALETTE = {
    0: "gray",
    1: "black",
    2: "tab:orange",
    3: "skyblue",
    4: "tab:green",
    5: "#C9A000",
    6: "tab:blue",
    7: "tab:red",
    8: "tab:purple",
}


def filter_rows_above_threshold(rows, threshold):
    return [
        row for row in rows
        if np.any(row["values"] > threshold)
    ]


def grouped_rows(rows, mode):
    groups = defaultdict(list)

    for row in rows:
        s = row["simple_name"]

        if mode == "amide_bond_count":
            k = amide_bond_count_from_simple_name(s)
        elif mode == "type1_symbol_count":
            k = type1_symbol_count_from_simple_name(s)
        else:
            raise ValueError("Unknown grouping mode.")

        groups[k].append(row)

    return groups


def plot_single_csv_grouped(
    filename,
    ylabel,
    output_svg,
    mode,
    threshold=None,
):
    _, rows, times = read_oligomer_csv(filename)

    if threshold is not None:
        rows = filter_rows_above_threshold(rows, threshold)

    groups = grouped_rows(rows, mode)

    if mode == "amide_bond_count":
        legend_title = "Amide bonds"
        label_suffix = "amide bonds"
        title_suffix = "by amide bond count"
    else:
        legend_title = "Type-1 symbols"
        label_suffix = "type-1 symbols"
        title_suffix = "by type-1 symbol count"

    threshold_text = "" if threshold is None else f" above threshold {threshold:g}"

    plt.figure(figsize=(8, 5))

    any_plotted = False

    for k in sorted(groups.keys()):
        color = PALETTE.get(k, None)

        for i, row in enumerate(groups[k]):
            plt.plot(
                times,
                row["values"],
                "-o",
                linewidth=1.2,
                markersize=4,
                alpha=0.35,
                color=color,
                label=f"{k} {label_suffix}" if i == 0 else None,
            )

        any_plotted = True

    plt.xlabel("Time days")
    plt.ylabel(ylabel)
    plt.title(f"{os.path.basename(filename)} — {title_suffix}{threshold_text}")

    if any_plotted:
        plt.legend(title=legend_title, frameon=False)
    else:
        print(f"No species passed threshold for {filename}")

    plt.tight_layout()
    plt.savefig(output_svg)
    plt.show()

    print(f"Saved plot: {output_svg}")


def plot_combined_normalized_grouped(
    pert_csv,
    nopert_csv,
    output_svg,
    mode,
    threshold_csv=None,
    threshold=None,
):
    _, pert_rows, times = read_oligomer_csv(pert_csv)
    _, nopert_rows, times2 = read_oligomer_csv(nopert_csv)

    if not np.allclose(times, times2):
        raise ValueError("Perturbed and nonperturbed CSV files have different time points.")

    pert_dict = {row["species"]: row for row in pert_rows}
    nopert_dict = {row["species"]: row for row in nopert_rows}

    shared_species = sorted(
        set(pert_dict.keys()) & set(nopert_dict.keys()),
        key=sort_key,
    )

    if threshold is not None:
        if threshold_csv is None:
            threshold_csv = pert_csv

        _, threshold_rows, _ = read_oligomer_csv(threshold_csv)
        threshold_dict = {row["species"]: row for row in threshold_rows}

        shared_species = [
            species for species in shared_species
            if species in threshold_dict
            and np.any(threshold_dict[species]["values"] > threshold)
        ]

    if mode == "amide_bond_count":
        count_func = amide_bond_count_from_simple_name
        legend_title = "Amide bonds"
        label_suffix = "amide bonds"
        title_suffix = "combined normalized by amide bond count"
    else:
        count_func = type1_symbol_count_from_simple_name
        legend_title = "Type-1 symbols"
        label_suffix = "type-1 symbols"
        title_suffix = "combined normalized by type-1 symbol count"

    groups = defaultdict(list)

    for species in shared_species:
        s = simple_name(species)
        k = count_func(s)
        groups[k].append(species)

    threshold_text = "" if threshold is None else f" above threshold {threshold:g}"

    plt.figure(figsize=(8, 5))

    any_plotted = False

    for k in sorted(groups.keys()):
        color = PALETTE.get(k, None)

        for i, species in enumerate(groups[k]):
            plt.plot(
                times,
                pert_dict[species]["values"],
                "-o",
                linewidth=1.2,
                markersize=4,
                alpha=0.35,
                color=color,
                label=f"{k} {label_suffix}" if i == 0 else None,
            )

            plt.plot(
                times,
                nopert_dict[species]["values"],
                "--o",
                linewidth=1.2,
                markersize=4,
                alpha=0.35,
                color=color,
            )

        any_plotted = True

    plt.xlabel("Time days")
    plt.ylabel("Min-max normalized concentration")
    plt.title(f"{title_suffix}{threshold_text}")

    if any_plotted:
        group_handles, group_labels = plt.gca().get_legend_handles_labels()

        style_handles = [
            plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5),
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5),
        ]

        first_legend = plt.legend(
            handles=group_handles,
            labels=group_labels,
            title=legend_title,
            frameon=False,
            loc="upper right",
        )

        plt.gca().add_artist(first_legend)

        plt.legend(
            handles=style_handles,
            labels=["Perturbed", "Nonperturbed"],
            frameon=False,
            loc="upper left",
        )
    else:
        print(f"No shared species passed threshold for combined plot: {output_svg}")

    plt.tight_layout()
    plt.savefig(output_svg)
    plt.show()

    print(f"Saved combined plot: {output_svg}")

def plot_combined_normalized_split_by_group(
    pert_csv,
    nopert_csv,
    output_prefix,
    mode,
    threshold_csv=None,
    threshold=None,
):
    _, pert_rows, times = read_oligomer_csv(pert_csv)
    _, nopert_rows, times2 = read_oligomer_csv(nopert_csv)

    if not np.allclose(times, times2):
        raise ValueError("Perturbed and nonperturbed CSV files have different time points.")

    pert_dict = {row["species"]: row for row in pert_rows}
    nopert_dict = {row["species"]: row for row in nopert_rows}

    shared_species = sorted(
        set(pert_dict.keys()) & set(nopert_dict.keys()),
        key=sort_key,
    )

    if threshold is not None:
        if threshold_csv is None:
            threshold_csv = pert_csv

        _, threshold_rows, _ = read_oligomer_csv(threshold_csv)
        threshold_dict = {row["species"]: row for row in threshold_rows}

        shared_species = [
            species for species in shared_species
            if species in threshold_dict
            and np.any(threshold_dict[species]["values"] > threshold)
        ]

    if mode == "amide_bond_count":
        count_func = amide_bond_count_from_simple_name
        label_name = "amide bonds"
        filename_name = "amide_count"
    elif mode == "type1_symbol_count":
        count_func = type1_symbol_count_from_simple_name
        label_name = "type-1 symbols"
        filename_name = "type1_count"
    else:
        raise ValueError("mode must be 'amide_bond_count' or 'type1_symbol_count'")

    groups = defaultdict(list)

    for species in shared_species:
        s = simple_name(species)
        k = count_func(s)
        groups[k].append(species)

    for k in sorted(groups.keys(), reverse=True):
        plt.figure(figsize=(8, 5))

        for species in groups[k]:
            pert_values = pert_dict[species]["values"]
            nopert_values = nopert_dict[species]["values"]

            plt.plot(
                times,
                pert_values,
                "-o",
                linewidth=1.2,
                markersize=4,
                alpha=0.45,
                color="black",
            )

            plt.plot(
                times,
                nopert_values,
                "--o",
                linewidth=1.2,
                markersize=4,
                alpha=0.45,
                color="tab:red",
            )

        plt.xlabel("Time days")
        plt.ylabel("Min-max normalized concentration")

        threshold_text = "" if threshold is None else f" above threshold {threshold:g}"

        plt.title(
            f"{OLIGOMER_LENGTH}mers with {k} {label_name}{threshold_text}"
        )

        style_handles = [
            plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5),
            plt.Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.5),
        ]

        plt.legend(
            handles=style_handles,
            labels=["Perturbed", "Nonperturbed"],
            frameon=False,
        )

        plt.tight_layout()

        output_svg = f"{output_prefix}_{filename_name}_{k}.svg"
        plt.savefig(output_svg)
        plt.show()

        print(f"Saved split combined plot: {output_svg}")


def plot_all_outputs():
    # Raw plots
    plot_single_csv_grouped(
        RAW_PERT_CSV,
        "Raw concentration",
        f"time_evolution_{OLIGOMER_LENGTH}mer_pert_raw_by_amide_count.svg",
        "amide_bond_count",
    )

    plot_single_csv_grouped(
        RAW_NOPERT_CSV,
        "Raw concentration",
        f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_raw_by_amide_count.svg",
        "amide_bond_count",
    )

    plot_single_csv_grouped(
        RAW_PERT_CSV,
        "Raw concentration",
        f"time_evolution_{OLIGOMER_LENGTH}mer_pert_raw_by_type1_count.svg",
        "type1_symbol_count",
    )

    plot_single_csv_grouped(
        RAW_NOPERT_CSV,
        "Raw concentration",
        f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_raw_by_type1_count.svg",
        "type1_symbol_count",
    )

    # Normalized plots
#     plot_single_csv_grouped(
#         NORM_PERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_normalized_by_amide_count.svg",
#         "amide_bond_count",
#     )
# 
#     plot_single_csv_grouped(
#         NORM_NOPERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_normalized_by_amide_count.svg",
#         "amide_bond_count",
#     )
# 
#     plot_single_csv_grouped(
#         NORM_PERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_normalized_by_type1_count.svg",
#         "type1_symbol_count",
#     )
# 
#     plot_single_csv_grouped(
#         NORM_NOPERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_normalized_by_type1_count.svg",
#         "type1_symbol_count",
#     )

    # Thresholded normalized plots
#     plot_single_csv_grouped(
#         RAW_PERT_CSV,
#         "Raw concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_raw_above_threshold_by_amide_count.svg",
#         "amide_bond_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         RAW_NOPERT_CSV,
#         "Raw concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_raw_above_threshold_by_amide_count.svg",
#         "amide_bond_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         RAW_PERT_CSV,
#         "Raw concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_raw_above_threshold_by_type1_count.svg",
#         "type1_symbol_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         RAW_NOPERT_CSV,
#         "Raw concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_raw_above_threshold_by_type1_count.svg",
#         "type1_symbol_count",
#         threshold=THRESHOLD,
#     )

#     plot_single_csv_grouped(
#         NORM_PERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_normalized_above_threshold_by_amide_count.svg",
#         "amide_bond_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         NORM_NOPERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_normalized_above_threshold_by_amide_count.svg",
#         "amide_bond_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         NORM_PERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_pert_normalized_above_threshold_by_type1_count.svg",
#         "type1_symbol_count",
#         threshold=THRESHOLD,
#     )
# 
#     plot_single_csv_grouped(
#         NORM_NOPERT_CSV,
#         "Min-max normalized concentration",
#         f"time_evolution_{OLIGOMER_LENGTH}mer_Nopert_normalized_above_threshold_by_type1_count.svg",
#         "type1_symbol_count",
#         threshold=THRESHOLD,
#     )

    # Combined normalized plots
    plot_combined_normalized_grouped(
        NORM_PERT_CSV,
        NORM_NOPERT_CSV,
        f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_by_amide_count.svg",
        "amide_bond_count",
    )

    plot_combined_normalized_grouped(
        NORM_PERT_CSV,
        NORM_NOPERT_CSV,
        f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_by_type1_count.svg",
        "type1_symbol_count",
    )

    # Combined normalized thresholded plots
    # Threshold is evaluated on RAW perturbed values.
    plot_combined_normalized_grouped(
        NORM_PERT_CSV,
        NORM_NOPERT_CSV,
        f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_above_threshold_by_amide_count.svg",
        "amide_bond_count",
        threshold_csv=RAW_PERT_CSV,
        threshold=THRESHOLD,
    )

    plot_combined_normalized_grouped(
        NORM_PERT_CSV,
        NORM_NOPERT_CSV,
        f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_above_threshold_by_type1_count.svg",
        "type1_symbol_count",
        threshold_csv=RAW_PERT_CSV,
        threshold=THRESHOLD,
    )

    # Split version of the last plot:
    # one separate plot for each type-1-symbol count.
    plot_combined_normalized_split_by_group(
        pert_csv=NORM_PERT_CSV,
        nopert_csv=NORM_NOPERT_CSV,
        output_prefix=f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_above_threshold",
        mode="type1_symbol_count",
        threshold_csv=RAW_PERT_CSV,
        threshold=THRESHOLD,
    )

    # Same idea, but split by amide bond count.
#     plot_combined_normalized_split_by_group(
#         pert_csv=NORM_PERT_CSV,
#         nopert_csv=NORM_NOPERT_CSV,
#         output_prefix=f"time_evolution_{OLIGOMER_LENGTH}mer_combined_normalized_above_threshold",
#         mode="amide_bond_count",
#         threshold_csv=RAW_PERT_CSV,
#         threshold=THRESHOLD,
#     )

# =============================================================================
# Main
# =============================================================================

def main():
    export_raw_oligomer_csv(
        input_file=PERTURBED_FILE,
        output_csv=RAW_PERT_CSV,
        oligomer_length=OLIGOMER_LENGTH,
        targets=TARGETS,
    )

    export_raw_oligomer_csv(
        input_file=NONPERTURBED_FILE,
        output_csv=RAW_NOPERT_CSV,
        oligomer_length=OLIGOMER_LENGTH,
        targets=TARGETS,
    )

    normalize_to_perturbed_minmax(
        input_csv=RAW_PERT_CSV,
        output_csv=NORM_PERT_CSV,
        reference_csv=RAW_PERT_CSV,
    )

    normalize_to_perturbed_minmax(
        input_csv=RAW_NOPERT_CSV,
        output_csv=NORM_NOPERT_CSV,
        reference_csv=RAW_PERT_CSV,
    )

    plot_all_outputs()

    print("\nDone.")


if __name__ == "__main__":
    main()