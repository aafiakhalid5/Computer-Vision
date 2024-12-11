import os
import subprocess
from itertools import product

gamma_values = [1, 10, 100]
C_values = [10, 100, 1000]
n_clusters_values = [50, 100, 200]
max_descriptors_values = [100000, 500000, 1000000]
use_powernorm = [True, False]
use_gmp = [True, False]

results_file = "exercise3.txt"

def delete_cached_files():
    """Deletes cached `.pkl.gz` files to ensure fresh runs."""
    for file in os.listdir():
        if file.endswith(".pkl.gz"):
            os.remove(file)
    print("> Deleted cached .pkl.gz files")

def run_experiment(params):
    """Runs the main script with the given parameters and logs results."""
    cmd = [
        "python", "exercise3.py",
        "--in_train", "train",
        "--in_test", "test",
        "--labels_train", "labels_train.txt",
        "--labels_test", "labels_test.txt",
        "--suffix", "_SIFT_patch_pr.pkl.gz",
        "--gamma", str(params["gamma"]),
        "--C", str(params["C"]),
        "--n_clusters", str(params["n_clusters"]),
        "--max_descriptors", str(params["max_descriptors"]),
    ]

    if params["powernorm"]:
        cmd.append("--powernorm")
    if params["gmp"]:
        cmd.append("--gmp")

    try:
        print(f"> Running experiment with params: {params}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        result.check_returncode()

        with open(results_file, "a") as f:
            f.write(f"Experiment Params: {params}\n")
            f.write(result.stdout)
            f.write("\n" + "-" * 50 + "\n")
        print("> Experiment completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"> Experiment failed with params: {params}")
        with open(results_file, "a") as f:
            f.write(f"Experiment Params: {params}\n")
            f.write(f"Error: {e.stderr}\n")
            f.write("\n" + "-" * 50 + "\n")

if __name__ == "__main__":

    experiments = list(product(
        gamma_values,
        C_values,
        n_clusters_values,
        max_descriptors_values,
        use_powernorm,
        use_gmp,
    ))

    for gamma, C, n_clusters, max_descriptors, powernorm, gmp in experiments:
        params = {
            "gamma": gamma,
            "C": C,
            "n_clusters": n_clusters,
            "max_descriptors": max_descriptors,
            "powernorm": powernorm,
            "gmp": gmp,
        }
        delete_cached_files()
        run_experiment(params)
