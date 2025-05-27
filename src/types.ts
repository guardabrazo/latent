export interface EmbeddingItem {
    filename: string;
    tsne: [number, number, number];
    pca: [number, number, number];
    cluster_kmeans: number;
    cluster_dbscan: number;
    cluster_agglom: number;
}

export type LayoutAlgorithm = 'tsne' | 'pca' | 'kmeans' | 'dbscan' | 'agglom';
