import pandas as pd

from src.data_preprocessing import preprocess_data
from src.rfm_analysis import create_rfm
from src.clustering import (
    find_optimal_clusters,
    perform_clustering
)
from src.pca_analysis import apply_pca
from src.visualization import (
    plot_elbow,
    plot_clusters,
    plot_pca_clusters
)


def main():

    file_path = "dataset/Mall_Customers.csv"

    df, scaled_data = preprocess_data(file_path)

    rfm = create_rfm(df)

    inertia = find_optimal_clusters(scaled_data)

    plot_elbow(inertia)

    labels, model = perform_clustering(
        scaled_data,
        n_clusters=5
    )

    df["Cluster"] = labels

    plot_clusters(df)

    pca_data = apply_pca(scaled_data)

    plot_pca_clusters(
        pca_data,
        labels
    )

    report = (
        df.groupby("Cluster")
        .mean(numeric_only=True)
    )

    report.to_csv(
        "outputs/cluster_report.csv"
    )

    print("\nCustomer Segmentation Completed!")
    print("\nCluster Summary:")
    print(report)


if __name__ == "__main__":
    main()