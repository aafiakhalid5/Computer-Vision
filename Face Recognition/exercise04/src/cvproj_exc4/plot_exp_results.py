import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config

def plot_experiment_results(filepath):
    df = pd.read_csv(filepath)
    df["Inertia Values"] = df["Inertia Values"].apply(
        lambda x: [float(val) for val in x.strip("[]").split(", ")]
    )

    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        inertia_values = row["Inertia Values"]
        num_clusters = row['Num Clusters']
        max_iter = row['Max Iter']
        seed = row['Random Seed']
        label = "Seed={}".format(seed)
        plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o', label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Inertia (Objective Function)")
    plt.title("Convergence of k-means Clustering")
    plt.legend(title="Seed")
    plt.grid(True)
    plt.show()

def main():
    plot_experiment_results(Config.clustering_experiments_result_file)

if __name__ == "__main__":
    main()
