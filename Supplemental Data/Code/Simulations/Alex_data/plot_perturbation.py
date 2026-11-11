import re
import ast
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams["svg.fonttype"] = "none"
SECONDS_PER_DAY = 24 * 3600


##########################################################################################
######################################## READ DATA #######################################
##########################################################################################

def simple_name(label):
	return label.replace("M_", "").replace("P_", "").replace("_", "")


def sort_key(label):
	s = simple_name(label)
	return (len(s), [int(c) for c in s])


def safe_minmax_normalize(x):
	x = np.asarray(x, dtype=float)
	vmin = float(np.min(x))
	vmax = float(np.max(x))

	if vmax == vmin:
		return np.zeros_like(x, dtype=float), True

	return (x - vmin) / (vmax - vmin), False


def species_color(species):
	length = len(simple_name(species))

	color_map = {
		1: "red",
		2: "blue",
		3: "green",
		4: "orange",
		5: "cyan",
		6: "lime",
		7: "purple",
	}

	return color_map.get(length, "gray")


def threshold_label(threshold_value):
	"""Return a compact label for filenames, titles and terminal output."""
	if isinstance(threshold_value, (list, tuple, np.ndarray)):
		return "[" + ", ".join(f"{float(x):.1e}" for x in threshold_value) + "]"

	return f"{float(threshold_value):.1e}"


def threshold_for_length(length, threshold_value):
	"""
	Return the threshold to use for an oligomer of encoded length `length`.

	If threshold_value is a scalar, the same threshold is used for all lengths.
	If threshold_value is a list/tuple/array, element 0 is used for monomers,
	element 1 for dimers, element 2 for trimers, etc.

	Example:
		length_dependent_threshold = [1e-3, 1e-4, 1e-5, 1e-6]

	means:
		length 1 -> 1e-3
		length 2 -> 1e-4
		length 3 -> 1e-5
		length 4 and longer -> 1e-6

	For lengths longer than the list, the last threshold is reused.
	This is usually what you want if you only define thresholds up to tetramers
	but your simulation also contains longer oligomers.
	"""
	if isinstance(threshold_value, (list, tuple, np.ndarray)):
		if len(threshold_value) == 0:
			raise ValueError("Length-dependent threshold list cannot be empty.")

		idx = min(int(length) - 1, len(threshold_value) - 1)
		return float(threshold_value[idx])

	return float(threshold_value)


def threshold_for_species(species, threshold_value):
	"""Threshold for a species, based on len(simple_name(species))."""
	return threshold_for_length(len(simple_name(species)), threshold_value)


def above_threshold(arr, species, threshold_value):
	"""Boolean array: concentration of this species is above its threshold."""
	thr = threshold_for_species(species, threshold_value)
	return np.asarray(arr, dtype=float) > thr


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


def parse_value(text):
	text = text.strip()

	try:
		return ast.literal_eval(text)
	except Exception:
		try:
			return float(text)
		except Exception:
			return text


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

			t_phys = np.array(ast.literal_eval(block), dtype=float)
			data["time"] = t_phys.tolist()
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
		raise ValueError("Could not find 't_phys_s:' in the new dict-like file.")

	n = len(data["time"])

	for k, v in data.items():
		if len(v) != n:
			raise ValueError(f"Column '{k}' has {len(v)} values, but time has {n}.")

	t_phys_arr = np.asarray(data["time"], dtype=float)

	W_v = np.array(
		[water_concentration_from_params(t, params) for t in t_phys_arr],
		dtype=float,
	)

	data["W_v"] = W_v.tolist()

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
	data_lines = lines[header_idx + 1:]

	data = {h: [] for h in headers}

	for line in data_lines:
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


##########################################################################################
######################################## PLOTTING ########################################
##########################################################################################

def read_and_plot(filename):
	data, params = read_timeseries(filename)

	time = np.asarray(data["time"], dtype=float) / SECONDS_PER_DAY
	W_v = np.asarray(data["W_v"], dtype=float)

	polymer_species = [h for h in data.keys() if h not in ("time", "W_v")]
	ordered_species = sorted(polymer_species, key=sort_key)

	print(f"Read {len(time)} time points.")
	print(f"Read {len(ordered_species)} species.")

	if params:
		print("Detected new dict-like output format.")

	######################################################################################
	############################### Plot Concentrations ##################################
	######################################################################################

	plt.figure(figsize=(10, 5))

	for h in ordered_species:
		plt.plot(
			time,
			data[h],
			c=species_color(h),
			label=h,
			linewidth=1.5,
		)

	plt.xlabel("Time (days)")
	plt.ylabel("Concentration")
	plt.title("Evolution of concentrations over time")
	plt.tight_layout()
	plt.show()

	######################################################################################
	############################## Plot Number of mols ####################################
	######################################################################################

	fig, ax1 = plt.subplots(figsize=(10, 5))

	for h in ordered_species:
		arr = np.asarray(data[h], dtype=float)
		data_mols = arr * W_v

		ax1.plot(
			time,
			data_mols,
			c=species_color(h),
			label=h,
			linewidth=1.5,
		)

	ax1.set_xlabel("Time (days)")
	ax1.set_ylabel("Number mols")
	ax1.set_title("Evolution of # mols over time")

	ax2 = ax1.twinx()
	ax2.plot(time, W_v, c="black", label="Water [W_v]", linewidth=1.0)
	ax2.set_ylabel("[Water]")

	plt.tight_layout()
	plt.show()

	######################################################################################
	######################## Plot Concentrations Normalized ###############################
	######################################################################################

	selected_times = np.linspace(0.99, 9.99, 10)
	indices = [np.argmin(np.abs(time - t)) for t in selected_times]

	plt.figure(figsize=(10, 5))

	for h in ordered_species:
		arr = np.asarray(data[h], dtype=float)
		selected_values = arr[indices]

		normalized, was_flat = safe_minmax_normalize(selected_values)

		if len(simple_name(h)) == 1:
			init_val = 1.0
		else:
			init_val = 0.0

		normalized = np.insert(normalized, 0, init_val)
		times_with_init = np.insert(time[indices], 0, 0.0)

		color = species_color(h)

		plt.plot(
			times_with_init,
			normalized,
			c=color,
			linewidth=1.5,
		)

		plt.text(
			times_with_init[-1] + 0.1,
			normalized[-1],
			h,
			color=color,
			fontsize=8,
			va="center",
		)

	plt.xlabel("Time (days)")
	plt.ylabel("Normalized concentration")
	plt.title("Evolution normalized over selected times")
	plt.tight_layout()
	plt.show()

	######################################################################################
	######################### SHARED TARGET OLIGOMER SET ##################################
	######################################################################################

# 	TARGETS = np.array([0.01, 0.99, 2.99, 5.99, 9.99], dtype=float)
# 	idx_targets = np.argmin(np.abs(time[:, None] - TARGETS[None, :]), axis=0)
# 	plot_times = time[idx_targets]
# 
# 	# Important:
# 	# plot_length is the number of bonds.
# 	# oligomer_length is the number of monomer units / encoded species length.
# 	#
# 	# If plot_length = 5, this selects length-6 species.
# 	# If you want heptamers, use plot_length = 6 at the bottom of the file.
# 	oligomer_length = plot_length + 1
# 
# 	target_species = []
# 
# 	for h in ordered_species:
# 		s = simple_name(h)
# 
# 		if len(s) != oligomer_length:
# 			continue
# 
# 		if not set(s) <= {"0", "1"}:
# 			continue
# 
# 		target_species.append(h)
# 
# 	print(f"Selected {len(target_species)} target species with encoded length {oligomer_length}.")
# 	print(f"These are length-{oligomer_length} species. If you wanted heptamers, set plot_length = 6.")
# 
# 	######################################################################################
# 	######################### GROUP BY NUMBER OF AMIDE BONDS ##############################
# 	######################################################################################
# 
# 	def extract_bond_pattern(name):
# 		s = simple_name(name)
# 
# 		if len(s) <= 1:
# 			return ""
# 
# 		return s[1:]
# 
# 	groups = defaultdict(list)
# 	groups_detected = defaultdict(list)
# 	groups_detected_length_threshold = defaultdict(list)
# 
# 	for h in target_species:
# 		arr = np.asarray(data[h], dtype=float)
# 		s = simple_name(h)
# 
# 		bond_pattern = extract_bond_pattern(h)
# 
# 		if len(bond_pattern) != plot_length:
# 			continue
# 
# 		amide_count = bond_pattern.count("1")
# 
# 		if 1 <= amide_count <= plot_length:
# 			sel = arr[idx_targets]
# 			norm, was_flat = safe_minmax_normalize(sel)
# 
# 			groups[amide_count].append(norm)
# 
# 			if np.any(arr > threshold):
# 				groups_detected[amide_count].append(norm)
# 
# 			if np.any(above_threshold(arr, h, length_dependent_threshold)):
# 				groups_detected_length_threshold[amide_count].append(norm)
# 
# 	palette = {
# 		0: "gray",
# 		1: "black",
# 		2: "tab:orange",
# 		3: "skyblue",
# 		4: "tab:green",
# 		5: "#C9A000",
# 		6: "tab:blue",
# 		7: "tab:red",
# 	}
# 
# 	ERROR_METRIC = "std"
# 
# 	def compute_error(mat, how="sem"):
# 		if mat.shape[0] == 1:
# 			return np.zeros(mat.shape[1], dtype=float)
# 
# 		if how == "std":
# 			return np.nanstd(mat, axis=0, ddof=1)
# 
# 		return np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
# 
# 	######################################################################################
# 	######################### Mean target oligomers by amide count #########################
# 	######################################################################################
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(1, plot_length + 1):
# 		traces = groups.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		mat = np.vstack(traces)
# 		mean_trace = np.nanmean(mat, axis=0)
# 		err = compute_error(mat, ERROR_METRIC)
# 
# 		plt.errorbar(
# 			plot_times,
# 			mean_trace,
# 			yerr=err,
# 			fmt="-o",
# 			linewidth=2,
# 			capsize=3,
# 			elinewidth=1.5,
# 			color=palette.get(k, None),
# 			label=f"{k} amide bonds (±{ERROR_METRIC.upper()})",
# 		)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel(f"Mean normalized concentration ± {ERROR_METRIC.upper()}")
# 	plt.title(f"Length-{oligomer_length} oligomers: mean normalized traces by amide count")
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers found for the mean amide-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_error_bar.svg")
# 	plt.show()
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by amide count
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(1, plot_length + 1):
# 		traces = groups.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} amide bonds" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(f"Length-{oligomer_length} oligomers: all normalized traces by amide count")
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers found for the all-amide-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amide_count.svg")
# 	plt.show()
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by amide count,
# 	# only species above threshold in raw/non-normalized data
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(1, plot_length + 1):
# 		traces = groups_detected.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} amide bonds" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(f"Length-{oligomer_length} oligomers above threshold: by amide count")
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers above threshold found for the amide-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amide_count_threshold_0,00001.svg")
# 	plt.show()
# 
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by amide count,
# 	# only species above length-dependent threshold in raw/non-normalized data
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(1, plot_length + 1):
# 		traces = groups_detected_length_threshold.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} amide bonds" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(
# 		f"Length-{oligomer_length} oligomers above length-dependent threshold: "
# 		f"by amide count"
# 	)
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers above length-dependent threshold found for the amide-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amide_count_length_threshold.svg")
# 	plt.show()
# 
# 	######################################################################################
# 	######################### GROUP BY NUMBER OF TYPE-1 SYMBOLS ###########################
# 	######################################################################################
# 
# 	# This uses exactly the same target_species list as the amide-count plot.
# 	# Therefore this plot cannot accidentally switch from hexamers to pentamers.
# 	#
# 	# Note: this counts "1" symbols in the full encoded species string s.
# 	# If your encoding is not truly monomer identity at every position,
# 	# then this is "number of type-1 symbols", not necessarily true monomer composition.
# 
# 	composition_groups = defaultdict(list)
# 	composition_groups_detected = defaultdict(list)
# 	composition_groups_detected_length_threshold = defaultdict(list)
# 
# 	for h in target_species:
# 		arr = np.asarray(data[h], dtype=float)
# 		s = simple_name(h)
# 
# 		type_1_count = s.count("1")
# 
# 		sel = arr[idx_targets]
# 		norm, was_flat = safe_minmax_normalize(sel)
# 
# 		composition_groups[type_1_count].append(norm)
# 
# 		if np.any(arr > threshold):
# 			composition_groups_detected[type_1_count].append(norm)
# 
# 		if np.any(above_threshold(arr, h, length_dependent_threshold)):
# 			composition_groups_detected_length_threshold[type_1_count].append(norm)
# 
# 	composition_palette = palette
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by type-1 count
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(0, oligomer_length + 1):
# 		traces = composition_groups.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = composition_palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} type-1 symbols" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(f"Length-{oligomer_length} oligomers: all normalized traces by type-1 count")
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers found for the type-1-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amino_count.svg")
# 	plt.show()
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by type-1 count,
# 	# only species above threshold in raw/non-normalized data
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(0, oligomer_length + 1):
# 		traces = composition_groups_detected.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = composition_palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} type-1 symbols" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(f"Length-{oligomer_length} oligomers above threshold: by type-1 count")
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers above threshold found for the type-1-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amino_count_threshold_0,00001.svg")
# 	plt.show()
# 
# 
# 	# ------------------------------------------------------------
# 	# Plot: all individual target configurations grouped by type-1 count,
# 	# only species above length-dependent threshold in raw/non-normalized data
# 	# ------------------------------------------------------------
# 
# 	plt.figure(figsize=(8, 5))
# 
# 	any_plotted = False
# 
# 	for k in range(0, oligomer_length + 1):
# 		traces = composition_groups_detected_length_threshold.get(k, [])
# 
# 		if not traces:
# 			continue
# 
# 		color = composition_palette.get(k, None)
# 
# 		for i, trace in enumerate(traces):
# 			plt.plot(
# 				plot_times,
# 				trace,
# 				"-o",
# 				linewidth=1.2,
# 				markersize=4,
# 				alpha=0.35,
# 				color=color,
# 				label=f"{k} type-1 symbols" if i == 0 else None,
# 			)
# 
# 		any_plotted = True
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Normalized concentration")
# 	plt.title(
# 		f"Length-{oligomer_length} oligomers above length-dependent threshold: "
# 		f"by type-1 count"
# 	)
# 
# 	if any_plotted:
# 		plt.legend(title="Oligomer class", frameon=False)
# 	else:
# 		print("No target oligomers above length-dependent threshold found for the type-1-count plot.")
# 
# 	plt.tight_layout()
# 	plt.savefig("figure_1_simulation_amino_count_length_threshold.svg")
# 	plt.show()
# 
# 	######################################################################################
# 	############################ SAVE CSV FILES ###########################################
# 	######################################################################################
# 
# 	csv_file = "oligomer_all_traces_by_amide_count.csv"
# 
# 	with open(csv_file, "w", newline="") as f:
# 		writer = csv.writer(f)
# 
# 		header = ["amide_count", "trace_id"] + [f"t={t:.2f}" for t in plot_times]
# 		writer.writerow(header)
# 
# 		for k in range(1, plot_length + 1):
# 			traces = groups.get(k, [])
# 
# 			if not traces:
# 				continue
# 
# 			for i, trace in enumerate(traces):
# 				row = [k, i] + list(trace)
# 				writer.writerow(row)
# 
# 	print(f"Saved CSV to {csv_file}")
# 
# 	csv_file = "oligomer_all_traces_by_type1_count.csv"
# 
# 	with open(csv_file, "w", newline="") as f:
# 		writer = csv.writer(f)
# 
# 		header = ["type_1_count", "trace_id"] + [f"t={t:.2f}" for t in plot_times]
# 		writer.writerow(header)
# 
# 		for k in range(0, oligomer_length + 1):
# 			traces = composition_groups.get(k, [])
# 
# 			if not traces:
# 				continue
# 
# 			for i, trace in enumerate(traces):
# 				row = [k, i] + list(trace)
# 				writer.writerow(row)
# 
# 	print(f"Saved CSV to {csv_file}")

	######################################################################################
	############################ Plot BARS concentrations ################################
	######################################################################################

	def closest_idx(day):
		return int(np.argmin(np.abs(time - day)))

	idx_2 = closest_idx(2.0)
	idx_3 = closest_idx(3.0)
	idx_4 = closest_idx(4.0)
	idx_10 = closest_idx(9.99)

	counts_2 = [data[h][idx_2] for h in ordered_species]
	counts_3 = [data[h][idx_3] for h in ordered_species]
	counts_4 = [data[h][idx_4] for h in ordered_species]
	counts_10 = [data[h][idx_10] for h in ordered_species]

	labels_simple = [simple_name(h) for h in ordered_species]

	plt.figure(figsize=(12, 5))

	plt.bar(
		range(len(ordered_species)),
		counts_10,
		color="b",
		alpha=1.0,
		label="time = 10 days",
	)

	plt.bar(
		range(len(ordered_species)),
		counts_4,
		color="r",
		alpha=0.3,
		label="time = 4 days",
	)

	plt.bar(
		range(len(ordered_species)),
		counts_3,
		color="yellow",
		alpha=0.15,
		label="time = 3 days",
	)

	plt.bar(
		range(len(ordered_species)),
		counts_2,
		color="black",
		alpha=0.2,
		label="time = 2 days",
	)

	plt.xticks(
		range(0, len(ordered_species), 5),
		labels_simple[::5],
		rotation=90,
		fontsize=6,
	)

	plt.ylabel("Concentration")
	plt.xlabel("Oligomer configuration")
	plt.legend()
	plt.tight_layout()
	plt.show()

# 	######################################################################################
# 	######################### Evolution Concentration Bonds ###############################
# 	######################################################################################
# 
# 	max_bond_type = 0
# 
# 	for h in ordered_species:
# 		s = simple_name(h)
# 
# 		if len(s) > 1:
# 			max_bond_type = max(max_bond_type, max(int(ch) for ch in s[1:]))
# 
# 	bond_conc = {
# 		bt: np.zeros(len(time), dtype=float)
# 		for bt in range(max_bond_type + 1)
# 	}
# 
# 	for h in ordered_species:
# 		s = simple_name(h)
# 		arr_mols = np.asarray(data[h], dtype=float) * W_v
# 
# 		if len(s) > 1:
# 			bonds = s[1:]
# 
# 			for bt in bond_conc:
# 				n_bonds = bonds.count(str(bt))
# 				bond_conc[bt] += arr_mols * n_bonds
# 
# 	plt.figure(figsize=(10, 5))
# 
# 	for bt, vals in bond_conc.items():
# 		plt.plot(
# 			time,
# 			vals,
# 			linewidth=1.5,
# 			label=f"Bond type {bt}",
# 		)
# 
# 	plt.xlabel("Time (days)")
# 	plt.ylabel("Concentration of bonds")
# 	plt.title("Evolution of bond concentrations")
# 	plt.legend()
# 	plt.tight_layout()
# 	plt.show()

	######################################################################################
	######################### Evolution Species above threshold ###########################
	######################################################################################

	num_species_above = np.zeros(len(time), dtype=float)

	for i in range(len(time)):
		count = 0

		for h in ordered_species:
			data_mols = data[h][i]

			if data_mols > threshold:
				count += 1

		num_species_above[i] = count

	plt.figure(figsize=(8, 5))

	plt.plot(time, num_species_above, c="b")

	plt.xlabel("Time (days)")
	plt.ylabel(f"Number of species with concentration > {threshold}")
	plt.title("Number of oligomer species above threshold over time")
	plt.tight_layout()
	plt.show()

	target_times = np.arange(0, 11)

	print(target_times)

	time_subset = []
	num_species_subset = []

	for t_int in target_times:
		idx = np.searchsorted(time, t_int, side="left") - 1

		if idx < 0:
			idx = 0

		time_subset.append(time[idx])
		num_species_subset.append(num_species_above[idx])

	print("Species above threshold at days 0..10:")
	print(num_species_subset)

	plt.figure(figsize=(8, 5))

	plt.plot(
		time_subset,
		num_species_subset,
		c="b",
		marker="o",
	)

	plt.xlabel("Time (days)")
	plt.ylabel(f"Number of species with concentration > {threshold}")
	plt.title("Number of oligomer species above threshold over time")
	plt.tight_layout()
	plt.show()

	######################################################################################
	################ Evolution Species above length-dependent threshold ###################
	######################################################################################

	print("\nLength-dependent threshold definition:")
	print(length_dependent_threshold)

	max_species_length = max(len(simple_name(h)) for h in ordered_species)
	print("Threshold used by encoded oligomer length:")
	for L in range(1, max_species_length + 1):
		print(f"  length {L}: {threshold_for_length(L, length_dependent_threshold):.6g}")

	num_species_above_length_threshold = np.zeros(len(time), dtype=float)
	num_species_above_length_threshold_by_length = {
		L: np.zeros(len(time), dtype=float)
		for L in range(1, max_species_length + 1)
	}

	for i in range(len(time)):
		total_count = 0

		for h in ordered_species:
			L = len(simple_name(h))
			thr_L = threshold_for_length(L, length_dependent_threshold)
			conc = data[h][i]

			if conc > thr_L:
				total_count += 1
				num_species_above_length_threshold_by_length[L][i] += 1

		num_species_above_length_threshold[i] = total_count

	plt.figure(figsize=(8, 5))

	plt.plot(
		time,
		num_species_above_length_threshold,
		c="b",
		label="all lengths",
	)

	plt.xlabel("Time (days)")
	plt.ylabel("Number of species above length-dependent threshold")
	plt.title("Number of oligomer species above length-dependent threshold over time")
	plt.tight_layout()
	plt.savefig("species_above_length_dependent_threshold_over_time.svg")
	plt.show()

	plt.figure(figsize=(8, 5))

	any_plotted = False

	for L in range(1, max_species_length + 1):
		vals = num_species_above_length_threshold_by_length[L]

		if np.max(vals) <= 0:
			continue

		plt.plot(
			time,
			vals,
			linewidth=1.5,
			label=f"length {L}, threshold {threshold_for_length(L, length_dependent_threshold):.1e}",
		)

		any_plotted = True

	plt.xlabel("Time (days)")
	plt.ylabel("Number of species above threshold")
	plt.title("Species above length-dependent threshold, separated by length")

	if any_plotted:
		plt.legend(frameon=False, fontsize=8)
	else:
		print("No species above length-dependent threshold for any length.")

	plt.tight_layout()
	plt.savefig("species_above_length_dependent_threshold_by_length.svg")
	plt.show()

	time_subset_length_threshold = []
	num_species_subset_length_threshold = []
	by_length_subset = {L: [] for L in range(1, max_species_length + 1)}

	for t_int in target_times:
		idx = np.searchsorted(time, t_int, side="left") - 1

		if idx < 0:
			idx = 0

		time_subset_length_threshold.append(time[idx])
		num_species_subset_length_threshold.append(num_species_above_length_threshold[idx])

		for L in range(1, max_species_length + 1):
			by_length_subset[L].append(num_species_above_length_threshold_by_length[L][idx])

	print("\nSpecies above length-dependent threshold at days 0..10:")
	print(num_species_subset_length_threshold)

	print("\nTOTAL species above length-dependent threshold at days 0..10:")
	for day, t_actual, total in zip(target_times, time_subset_length_threshold, num_species_subset_length_threshold):
		print(f"day {day} (actual time {t_actual:.3f}): total = {int(total)}")

	print("\nSpecies above length-dependent threshold by length at days 0..10:")
	print("days:", list(target_times))
	for L in range(1, max_species_length + 1):
		vals = by_length_subset[L]
		if any(v > 0 for v in vals):
			print(
				f"length {L} "
				f"(threshold {threshold_for_length(L, length_dependent_threshold):.6g}): "
				f"{vals}"
			)

	plt.figure(figsize=(8, 5))

	plt.plot(
		time_subset_length_threshold,
		num_species_subset_length_threshold,
		c="b",
		marker="o",
	)

	plt.xlabel("Time (days)")
	plt.ylabel("Number of species above length-dependent threshold")
	plt.title("Number of oligomer species above length-dependent threshold over time")
	plt.tight_layout()
	plt.savefig("species_above_length_dependent_threshold_days_0_to_10.svg")
	plt.show()
	
	print ('TOTAL',num_species_subset_length_threshold)



##########################################################################################
#################################### NAME OF THE FILE ####################################
##########################################################################################

threshold = 0.001 * 1.0

# Length-dependent threshold.
# Element 0 is for monomers, element 1 for dimers, element 2 for trimers, etc.
# If the oligomer is longer than the list, the final value is reused.
# Example below: monomers 1e-3, dimers 1e-4, trimers 1e-5, tetramers and longer 1e-6.
length_dependent_threshold = [0.01,0.01, 0.005, 0.001, 0.0002]
length_dependent_threshold = [0.002,0.002, 0.001, 0.0002, 0.00004]
length_dependent_threshold = [0.005,0.005, 0.0025, 0.0005, 0.0001]

# This is the number of bonds.
# plot_length = 5 selects species of encoded length 6.
# plot_length = 6 selects species of encoded length 7.
plot_length = 5 - 1

read_and_plot("timeseries_thio-ester-amide_Nopert.txt")