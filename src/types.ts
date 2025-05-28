export interface EmbeddingItem {
    filename: string;
    tsne: [number, number, number];
    pca: [number, number, number];
    tsne_2d: [number, number];      // For 2D t-SNE scatter plot
    tsne_2d_grid: [number, number]; // Original grid-like tSNE data (might be sparse or used for another purpose)
    tsne_2d_grid_snap: [number, number]; // For the new semantic grid layout [col, row]
    cluster_kmeans: number;
    cluster_dbscan: number;
    cluster_agglom: number;
}

export type LayoutAlgorithm = 'tsne' | 'pca' | 'kmeans' | 'dbscan' | 'agglom';
// We might add new layout algorithm types later if needed for the new views,
// or handle them with a separate visualization mode state.
