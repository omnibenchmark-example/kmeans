import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans


def kmeans(distance_matrix, n_clusters, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=seed)
    kmeans.fit(distance_matrix)

    return kmeans.labels_


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Perform kmeans clustering to a distance matrix.')

    # Add arguments
    parser.add_argument('--distances', type=str, help='distance matrix on which to perform kmeans clustering', required=True)
    parser.add_argument('--clusters', type=int, help='number of clusters for kmeans', default=3)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--output_dir', type=str, help='output directory where kmeans cluster results will be saved.', default=os.getcwd())
    parser.add_argument('--name', type=str, help='name of the dataset', default='kmeans')

    # Parse arguments
    args = parser.parse_args()

    distances_df = pd.read_csv(args.distances)
    distance_matrix = distances_df.loc[:, distances_df.columns != 'id'].values

    clusters = kmeans(distance_matrix, args.clusters, args.seed)
    clusters_df = pd.DataFrame(clusters, index=distances_df['id'], columns=['label'])

    # Write distances to disk
    clusters_df.to_csv(Path(args.output_dir) / f'{args.name}.kmeans.csv', index_label='id')


if __name__ == "__main__":
    main()
