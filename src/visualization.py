import matplotlib.pyplot as plt


def plot_elbow(inertia):

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1,11),
        inertia,
        marker='o'
    )

    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.savefig("outputs/elbow_method.png")
    plt.close()


def plot_clusters(df):

    plt.figure(figsize=(8,5))

    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")

    plt.title("Customer Segments")

    plt.savefig("outputs/customer_clusters.png")
    plt.close()


def plot_pca_clusters(pca_data, labels):

    plt.figure(figsize=(8,5))

    plt.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=labels
    )

    plt.title("PCA Customer Segmentation")

    plt.savefig("outputs/pca_clusters.png")
    plt.close()