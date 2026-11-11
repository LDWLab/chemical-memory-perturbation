import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
t = np.arange(0, 11)

def safe_ratio(a, b):
    out = np.zeros_like(a, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    out[~mask] = np.nan
    return out

datasets = [

    {
        "name": "Data 6",
        "threshold": 0.0001,
        "thio_ester_amide": ([3.0, 13.0, 14.0, 162.0, 62.0, 52.0, 41.0, 39.0, 38.0, 36.0, 37.0],
                            [3.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
        "thio_ester": ([2.0, 8.0, 8.0, 57.0, 25.0, 21.0, 17.0, 15.0, 14.0, 14.0, 14.0],
                            [2.0, 8.0, 8.0, 8.0, 8.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0]),
        "ester_amino": ([2.0, 4.0, 4.0, 51.0, 42.0, 30.0, 22.0, 20.0, 18.0, 18.0, 15.0],
                            [2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        "thioester_amino": ([2.0, 7.0, 8.0, 31.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                            [2.0, 7.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
    },
    
    {
        "name": "Experimental",
        "threshold": 0.0001,
        "thio_ester_amide": ([1., 1.1, .9, 2.9, 2.4, 2.3, 2.1, 2.05, 2.3, 2.3, 2.3],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "thio_ester": ([1.4, 1.1, 1.3, 1.7, 1.1, .8, .8, 1.0, .9, .8, .7],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "ester_amino": ([.7, .6, .6, 1.9, 1.7, 1.4, 1.2, 1.2, 1.2, 1.2, 1.2],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "thioester_amino": ([1.2, .9, 1.2, 2.7, 2.9, 2., 1.8, 2.1, 2.0, 2.2, 2.3],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    
    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.0002,
        "thio_ester_amide": ([3.0, 6.0, 6.0, 75.0, 21.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                            [3.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0]),
        "thio_ester": ([2.0, 5.0, 6.0, 31.0, 17.0, 9.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [2.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 25.0, 14.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 3.0, 3.0, 27.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
                            [2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    },
    
    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.00004,
        "thio_ester_amide": ([3.0, 14.0, 18.0, 161.0, 130.0, 130.0, 121.0, 107.0, 99.0, 96.0, 94.0],
                            [3.0, 14.0, 18.0, 23.0, 26.0, 29.0, 33.0, 35.0, 38.0, 45.0, 51.0]),
        "thio_ester": ([2.0, 14.0, 26.0, 55.0, 52.0, 52.0, 53.0, 51.0, 48.0, 47.0, 47.0],
                            [2.0, 14.0, 26.0, 39.0, 44.0, 46.0, 46.0, 46.0, 47.0, 47.0, 47.0]),
        "ester_amino": ([2.0, 2.0, 3.0, 48.0, 33.0, 22.0, 13.0, 9.0, 7.0, 7.0, 7.0],
                            [2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        "thioester_amino": ([2.0, 10.0, 19.0, 31.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [2.0, 10.0, 19.0, 20.0, 27.0, 34.0, 34.0, 34.0, 34.0, 34.0, 33.0]),
    },

    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.0001,
        "thio_ester_amide": ([3.0, 7.0, 8.0, 116.0, 56.0, 33.0, 21.0, 15.0, 13.0, 12.0, 12.0],
                            [3.0, 7.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
        "thio_ester": ([2.0, 6.0, 6.0, 42.0, 34.0, 28.0, 20.0, 15.0, 13.0, 13.0, 12.0],
                            [2.0, 6.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 35.0, 27.0, 11.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 7.0, 6.0, 31.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
                            [2.0, 7.0, 6.0, 6.0, 8.0, 6.0, 8.0, 14.0, 22.0, 22.0, 23.0]),
    },

    {
        "name": "Dry 75%, Wet 25%, lenght-dependent threshold, 2-3 monomer dependent threshold",
        "threshold": 0.0002,
        "thio_ester_amide": ([3.0, 14.0, 18.0, 161.0, 130.0, 130.0, 121.0, 107.0, 99.0, 96.0, 94.0],
                            [3.0, 14.0, 18.0, 23.0, 26.0, 29.0, 33.0, 35.0, 38.0, 45.0, 51.0]),
        "thio_ester": ([2.0, 5.0, 6.0, 31.0, 17.0, 9.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [2.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 25.0, 14.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 3.0, 3.0, 27.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
                            [2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    },

]

labels = {
    "thio_ester_amide": "Thio–Ester–Amide",
    "thio_ester": "Thio–Ester",
    "ester_amino": "Ester–Amino",
    "thioester_amino": "Thioester–Amino",
}

colors = {
    "thio_ester_amide": "black",
    "thio_ester": "tab:blue",
    "ester_amino": "tab:green",
    "thioester_amino": "tab:red",
}

for ds in datasets:
    plt.figure(figsize=(8, 5))

    for key in labels:
        pert, no_pert = ds[key]
        pert = np.array(pert, dtype=float)
        no_pert = np.array(no_pert, dtype=float)

        ratio = safe_ratio(pert, no_pert)

        plt.plot(
            t,
            ratio,
            "-o",
            linewidth=2,
            label=labels[key],
            color=colors[key],
        )

    plt.axhline(1.0, linestyle="--", linewidth=1, color="gray")

    plt.xlabel("Time (days)")
    plt.ylabel("Perturbed / Non-perturbed")
    plt.title(f"{ds['name']} amplification ratio")

    plt.text(
        0.03,
        0.95,
        f"Threshold = {ds['threshold']}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=10,
    )

    plt.legend(frameon=False)
    plt.tight_layout()
    filename = ds["name"].lower().replace(" ", "_")
    plt.savefig(f"{filename}.svg", format="svg")
    plt.show()
    
########
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 11)

def safe_ratio(a, b):
    out = np.zeros_like(a, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    out[~mask] = np.nan
    return out

# Normalize: min → 0, max → 1
def normalize_0_1(x):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if xmax == xmin:
        return np.zeros_like(x)  # flat case

    #return (x/xmax)
    return ((x-1)/xmax)
    #return (x - xmin) / (xmax - xmin)
    #return (x - x[0]) / (xmax - x[0])


datasets = [

    {
        "name": "Data 6",
        "threshold": 0.0001,
        "thio_ester_amide": ([3.0, 13.0, 14.0, 162.0, 62.0, 52.0, 41.0, 39.0, 38.0, 36.0, 37.0],
                            [3.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
        "thio_ester": ([2.0, 8.0, 8.0, 57.0, 25.0, 21.0, 17.0, 15.0, 14.0, 14.0, 14.0],
                            [2.0, 8.0, 8.0, 8.0, 8.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0]),
        "ester_amino": ([2.0, 4.0, 4.0, 51.0, 42.0, 30.0, 22.0, 20.0, 18.0, 18.0, 15.0],
                            [2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        "thioester_amino": ([2.0, 7.0, 8.0, 31.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                            [2.0, 7.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
    },
    
    {
        "name": "Experimental",
        "threshold": 0.0001,
        "thio_ester_amide": ([1., 1.1, .9, 2.9, 2.4, 2.3, 2.1, 2.05, 2.3, 2.3, 2.3],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "thio_ester": ([1.4, 1.1, 1.3, 1.7, 1.1, .8, .8, 1.0, .9, .8, .7],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "ester_amino": ([.7, .6, .6, 1.9, 1.7, 1.4, 1.2, 1.2, 1.2, 1.2, 1.2],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "thioester_amino": ([1.2, .9, 1.2, 2.7, 2.9, 2., 1.8, 2.1, 2.0, 2.2, 2.3],
                            [1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    
    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.0002,
        "thio_ester_amide": ([3.0, 6.0, 6.0, 75.0, 21.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                            [3.0, 6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0]),
        "thio_ester": ([2.0, 5.0, 6.0, 31.0, 17.0, 9.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [2.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 25.0, 14.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 3.0, 3.0, 27.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
                            [2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    },
    
    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.00004,
        "thio_ester_amide": ([3.0, 14.0, 18.0, 161.0, 130.0, 130.0, 121.0, 107.0, 99.0, 96.0, 94.0],
                            [3.0, 14.0, 18.0, 23.0, 26.0, 29.0, 33.0, 35.0, 38.0, 45.0, 51.0]),
        "thio_ester": ([2.0, 14.0, 26.0, 55.0, 52.0, 52.0, 53.0, 51.0, 48.0, 47.0, 47.0],
                            [2.0, 14.0, 26.0, 39.0, 44.0, 46.0, 46.0, 46.0, 47.0, 47.0, 47.0]),
        "ester_amino": ([2.0, 2.0, 3.0, 48.0, 33.0, 22.0, 13.0, 9.0, 7.0, 7.0, 7.0],
                            [2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        "thioester_amino": ([2.0, 10.0, 19.0, 31.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [2.0, 10.0, 19.0, 20.0, 27.0, 34.0, 34.0, 34.0, 34.0, 34.0, 33.0]),
    },

    {
        "name": "dry 75%, wet 25%",
        "threshold": 0.0001,
        "thio_ester_amide": ([3.0, 7.0, 8.0, 116.0, 56.0, 33.0, 21.0, 15.0, 13.0, 12.0, 12.0],
                            [3.0, 7.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
        "thio_ester": ([2.0, 6.0, 6.0, 42.0, 34.0, 28.0, 20.0, 15.0, 13.0, 13.0, 12.0],
                            [2.0, 6.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 35.0, 27.0, 11.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 7.0, 6.0, 31.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
                            [2.0, 7.0, 6.0, 6.0, 8.0, 6.0, 8.0, 14.0, 22.0, 22.0, 23.0]),
    },

    {
        "name": "Dry 75%, Wet 25%, lenght-dependent threshold, 2-3 monomer dependent threshold",
        "threshold": 0.0002,
        "thio_ester_amide": ([3.0, 14.0, 18.0, 161.0, 130.0, 130.0, 121.0, 107.0, 99.0, 96.0, 94.0],
                            [3.0, 14.0, 18.0, 23.0, 26.0, 29.0, 33.0, 35.0, 38.0, 45.0, 51.0]),
        "thio_ester": ([2.0, 5.0, 6.0, 31.0, 17.0, 9.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [2.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        "ester_amino": ([2.0, 2.0, 2.0, 25.0, 14.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "thioester_amino": ([2.0, 3.0, 3.0, 27.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
                            [2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    },

]
labels = {
    "thio_ester_amide": "Thio–Ester–Amide",
    "thio_ester": "Thio–Ester",
    "ester_amino": "Ester–Amino",
    "thioester_amino": "Thioester–Amino",
}

colors = {
    "thio_ester_amide": "black",
    "thio_ester": "tab:blue",
    "ester_amino": "tab:green",
    "thioester_amino": "tab:red",
}

for ds in datasets:
    plt.figure(figsize=(8, 5))

    for key in labels:
        pert, no_pert = ds[key]
        pert = np.array(pert, dtype=float)
        no_pert = np.array(no_pert, dtype=float)

        ratio = safe_ratio(pert, no_pert)
        ratio_norm = normalize_0_1(ratio)

        plt.plot(
            t,
            ratio_norm,
            "-o",
            linewidth=2,
            label=labels[key],
            color=colors[key],
        )

    plt.axhline(.0, linestyle="--", linewidth=1, color="gray")
    plt.ylim(-0.25, 1)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized ratio (0–1)")
    plt.title(f"{ds['name']} (threshold = {ds['threshold']})")

    plt.legend(frameon=False)
    plt.tight_layout()

    filename = ds["name"].lower().replace(" ", "_") + "_normalized"
    plt.savefig(f"{filename}.svg", format="svg")

    plt.show()