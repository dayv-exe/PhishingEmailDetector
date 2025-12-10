import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings('ignore')

# ==================== ENHANCED STYLE CONFIGURATION ====================
# Modern color palette
COLORS = {
    'primary': '#2E86AB',  # Deep blue
    'secondary': '#A23B72',  # Purple
    'accent1': '#F18F01',  # Orange
    'accent2': '#C73E1D',  # Red
    'success': '#06A77D',  # Teal
    'warning': '#F4D35E',  # Yellow
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#F4D35E'],
    'gradient': ['#06A77D', '#2E86AB', '#A23B72', '#C73E1D'],
    'noise': '#95A3A4',  # Gray for noise points
    'background': '#F8F9FA'
}

# Set enhanced style
sns.set_style("whitegrid", {
    'grid.color': '.92',
    'grid.linestyle': '-',
    'axes.edgecolor': '.15',
    'axes.linewidth': 1.25
})
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.facecolor': 'white',
    'axes.facecolor': COLORS['background'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial']
})

# Create output directory
OUTPUT_DIR = 'outputs/clustering_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("CLUSTERING ANALYSIS FOR PHISHING EMAIL DETECTION")
print("=" * 80)

print("\n[STEP 1] Loading and Preprocessing Data")
print("-" * 80)

# Load data with row limit
MAX_ROWS = 10000
df = pd.read_csv('outputs/FeatureEngineered_PhishingEmailData.csv', nrows=MAX_ROWS)
print(f"Loaded {len(df)} samples with {len(df.columns)} features")
if MAX_ROWS is not None:
    print(f"(Limited to first {MAX_ROWS} rows)")

# ============== USER-SPECIFIED COLUMNS TO DROP ==============
COLUMNS_TO_DROP = [
    'month', 'time', 'year', 'top_level_domain_encoded',
    'second_level_domain_encoded', 'has_url', 'has_phone_number',
    'subject_typo_count', 'body_typo_count', 'subject_special_char_count',
    'body_special_char_count', 'has_non_ascii_chars', 'body_word_count',
    'body_uppercase_count', 'body_unique_word_count',
]

# Separate labels and features
y_true = df['label'].copy()
X = df.drop(columns=['label'])

# Drop user-specified columns if they exist
if COLUMNS_TO_DROP:
    existing_cols_to_drop = [col for col in COLUMNS_TO_DROP if col in X.columns]
    if existing_cols_to_drop:
        print(f"\nDropping {len(existing_cols_to_drop)} user-specified columns:")
        for col in existing_cols_to_drop:
            print(f"  • {col}")
        X = X.drop(columns=existing_cols_to_drop)
    missing_cols = [col for col in COLUMNS_TO_DROP if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: {len(missing_cols)} specified columns not found in dataset:")
        for col in missing_cols:
            print(f"  • {col}")
else:
    print("\nNo columns specified for removal")

print(f"Ground truth distribution: {y_true.value_counts().to_dict()}")

# Select numerical features only
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if 'hash' not in col.lower()]
print(f"Selected {len(numerical_cols)} numerical features")

# Create feature matrix and handle missing values
X_features = X[numerical_cols].fillna(X[numerical_cols].mean())
numerical_cols = None

# Feature Scaling
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_scaled = None

print(f"Reduced dimensions: {X_features.shape[1]} → {X_pca.shape[1]}")
print(f"Total variance preserved: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

# PCA variance visualisation
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('white')

ax1 = plt.subplot(1, 2, 1)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
ax1.plot(range(1, len(cumsum_variance) + 1), cumsum_variance,
         color=COLORS['primary'], linewidth=3, marker='o', markersize=6,
         markerfacecolor=COLORS['accent1'], markeredgewidth=2, markeredgecolor=COLORS['primary'])
ax1.axhline(y=0.95, color=COLORS['accent2'], linestyle='--', linewidth=2.5,
            label='95% variance threshold', alpha=0.8)
ax1.fill_between(range(1, len(cumsum_variance) + 1), cumsum_variance,
                 alpha=0.2, color=COLORS['primary'])
ax1.set_xlabel('Number of Components', fontweight='semibold')
ax1.set_ylabel('Cumulative Variance Explained', fontweight='semibold')
ax1.set_title('PCA Cumulative Variance', pad=15)
ax1.legend(frameon=True, shadow=True)
ax1.grid(True, alpha=0.25, linestyle='--')

ax2 = plt.subplot(1, 2, 2)
n_components = min(15, len(pca.explained_variance_ratio_))
bars = ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components],
               color=COLORS['gradient'][0], edgecolor=COLORS['gradient'][2], linewidth=1.5, alpha=0.85)
# Gradient coloring
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.viridis(i / n_components))
ax2.set_xlabel('Principal Component', fontweight='semibold')
ax2.set_ylabel('Variance Explained', fontweight='semibold')
ax2.set_title('Individual Component Variance (Top 15)', pad=15)
ax2.grid(True, alpha=0.25, axis='y', linestyle='--')

plt.tight_layout()
plt.show()
print("Displayed: PCA Variance plots")

# FIND OPTIMAL K
print("\n[STEP 2] Determining Optimal K")
print("-" * 80)

k_range = range(2, 11)
silhouette_scores = []
inertias = []

print("Testing k from 2 to 10...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, labels))
    inertias.append(kmeans.inertia_)
    print(f"k={k}: Silhouette={silhouette_scores[-1]:.3f}, Inertia={inertias[-1]:.2f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k = {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")

# Plot k selection with enhanced styling
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Silhouette plot
axes[0].plot(k_range, silhouette_scores, color=COLORS['primary'], linewidth=3.5,
             marker='o', markersize=10, markerfacecolor=COLORS['accent1'],
             markeredgewidth=2.5, markeredgecolor=COLORS['primary'], label='Silhouette Score')
axes[0].axvline(x=optimal_k, color=COLORS['accent2'], linestyle='--', linewidth=3,
                label=f'Optimal k={optimal_k}', alpha=0.8)
axes[0].scatter([optimal_k], [silhouette_scores[optimal_k - 2]],
                s=300, color=COLORS['accent2'], edgecolors='white', linewidths=3, zorder=5)
axes[0].set_xlabel('Number of Clusters (k)', fontweight='semibold')
axes[0].set_ylabel('Silhouette Score', fontweight='semibold')
axes[0].set_title('Silhouette Score vs k', pad=15)
axes[0].legend(frameon=True, shadow=True)
axes[0].grid(True, alpha=0.25, linestyle='--')

# Elbow plot
axes[1].plot(k_range, inertias, color=COLORS['success'], linewidth=3.5,
             marker='s', markersize=10, markerfacecolor=COLORS['warning'],
             markeredgewidth=2.5, markeredgecolor=COLORS['success'], label='Inertia')
axes[1].axvline(x=optimal_k, color=COLORS['accent2'], linestyle='--', linewidth=3,
                label=f'Selected k={optimal_k}', alpha=0.8)
axes[1].scatter([optimal_k], [inertias[optimal_k - 2]],
                s=300, color=COLORS['accent2'], edgecolors='white', linewidths=3, zorder=5)
axes[1].set_xlabel('Number of Clusters (k)', fontweight='semibold')
axes[1].set_ylabel('Inertia (WCSS)', fontweight='semibold')
axes[1].set_title('Elbow Method', pad=15)
axes[1].legend(frameon=True, shadow=True)
axes[1].grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()
plt.show()
print("Displayed: Optimal k selection plots")

# RUN CLUSTERING ALGORITHMS
print("\n[STEP 3] Running Clustering Algorithms")
print("-" * 80)

clustering_results = {}

# 1. K-MEANS
print("\n[3.1] K-Means Clustering")
print(f"Parameters: n_clusters={optimal_k}, random_state=42")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)
clustering_results['KMeans'] = labels_kmeans
print(f"Clusters: {len(np.unique(labels_kmeans))}")
print(f"Cluster sizes: {np.bincount(labels_kmeans)}")

# 2. DBSCAN
print("\n[3.2] DBSCAN Clustering")
print("Finding optimal epsilon using k-distance graph...")
k_neighbors = min(2 * X_pca.shape[1], len(X_pca) - 1)
neighbors = NearestNeighbors(n_neighbors=k_neighbors)
neighbors.fit(X_pca)
distances, _ = neighbors.kneighbors(X_pca)
sorted_distances = np.sort(distances[:, k_neighbors - 1])
eps = np.percentile(sorted_distances, 90)

print(f"Parameters: eps={eps:.3f}, min_samples={k_neighbors}")
dbscan = DBSCAN(eps=eps, min_samples=k_neighbors)
labels_dbscan = dbscan.fit_predict(X_pca)
clustering_results['DBSCAN'] = labels_dbscan

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)
print(f"Clusters: {n_clusters_dbscan}")
print(f"Noise points: {n_noise} ({n_noise / len(labels_dbscan) * 100:.1f}%)")

# 3. AGGLOMERATIVE CLUSTERING
print("\n[3.3] Agglomerative Hierarchical Clustering")
print(f"Parameters: n_clusters={optimal_k}, linkage='ward'")
agglomerative = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_agg = agglomerative.fit_predict(X_pca)
clustering_results['Agglomerative'] = labels_agg
print(f"Clusters: {len(np.unique(labels_agg))}")
print(f"Cluster sizes: {np.bincount(labels_agg)}")

# Create enhanced dendrogram
print("Creating dendrogram...")
linkage_matrix = linkage(X_pca, method='ward')

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor('white')
ax = plt.gca()
ax.set_facecolor(COLORS['background'])

dendro = dendrogram(linkage_matrix, truncate_mode='lastp', p=30,
                    color_threshold=linkage_matrix[-(optimal_k - 1), 2],
                    above_threshold_color=COLORS['noise'])
ax.set_xlabel('Cluster Size', fontweight='semibold', fontsize=12)
ax.set_ylabel('Distance', fontweight='semibold', fontsize=12)
ax.set_title('Agglomerative Clustering Dendrogram', pad=15, fontsize=14)
ax.axhline(y=linkage_matrix[-(optimal_k - 1), 2], color=COLORS['accent2'],
           linestyle='--', linewidth=2.5, label=f'Cut at k={optimal_k}', alpha=0.8)
ax.legend(frameon=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.25, axis='y', linestyle='--')

plt.tight_layout()
plt.show()
print("Displayed: Dendrogram")

# 4. SPECTRAL CLUSTERING
print("\n[3.4] Spectral Clustering")
if len(X_pca) > 2000:
    print(f"SKIPPED: Dataset too large (n={len(X_pca)} > 2000)")
    print("Reason: Spectral clustering has O(n³) complexity - would take too long")
else:
    print(f"Parameters: n_clusters={optimal_k}, affinity='rbf'")
    spectral = SpectralClustering(n_clusters=optimal_k, affinity='rbf',
                                  random_state=42, n_init=10)
    labels_spectral = spectral.fit_predict(X_pca)
    clustering_results['Spectral'] = labels_spectral
    print(f"Clusters: {len(np.unique(labels_spectral))}")
    print(f"Cluster sizes: {np.bincount(labels_spectral)}")

# EVALUATE CLUSTERING QUALITY
print("\n[STEP 4] Evaluating Clustering Quality")
print("-" * 80)

print("\nMetrics Explained:")
print("  • Silhouette Score [-1, 1]: Higher = better separation (>0.5 is good)")
print("  • Davies-Bouldin [0, ∞): Lower = better (compact & separated)")
print("  • Calinski-Harabasz [0, ∞): Higher = better (dense & separated)")
print("  • Adjusted Rand Index [-1, 1]: Similarity to ground truth (1 = perfect)")
print("  • Normalized Mutual Info [0, 1]: Information shared with ground truth")

evaluation_results = []

for algo_name, labels in clustering_results.items():
    print(f"\n{algo_name}:")

    if algo_name == 'DBSCAN':
        mask = labels >= 0
        X_eval = X_pca[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X_pca
        labels_eval = labels

    n_clusters = len(set(labels_eval))
    if n_clusters < 2:
        print(f"  Only {n_clusters} cluster - skipping evaluation")
        continue

    silhouette = silhouette_score(X_eval, labels_eval)
    davies_bouldin = davies_bouldin_score(X_eval, labels_eval)
    calinski = calinski_harabasz_score(X_eval, labels_eval)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print(f"  Silhouette Score:      {silhouette:.4f}")
    print(f"  Davies-Bouldin Index:  {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz:     {calinski:.2f}")
    print(f"  Adjusted Rand Index:   {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")

    evaluation_results.append({
        'Algorithm': algo_name,
        'N_Clusters': n_clusters,
        'N_Noise': list(labels).count(-1),
        'Silhouette': silhouette,
        'Davies_Bouldin': davies_bouldin,
        'Calinski_Harabasz': calinski,
        'ARI': ari,
        'NMI': nmi
    })

eval_df = pd.DataFrame(evaluation_results)

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(eval_df.to_string(index=False))

best_idx = eval_df['Silhouette'].idxmax()
best_algorithm = eval_df.loc[best_idx, 'Algorithm']
print(f"\n★ BEST ALGORITHM: {best_algorithm} (Silhouette: {eval_df.loc[best_idx, 'Silhouette']:.4f})")

eval_df.to_csv(f'{OUTPUT_DIR}/evaluation_metrics.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/evaluation_metrics.csv")

# Enhanced metrics comparison visualization
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('white')
fig.suptitle('Clustering Algorithm Comparison', fontsize=18, fontweight='bold', y=0.995)

metrics = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz', 'ARI', 'NMI']
metric_colors = [COLORS['primary'], COLORS['accent2'], COLORS['success'],
                 COLORS['accent1'], COLORS['secondary']]

for idx, (metric, color) in enumerate(zip(metrics, metric_colors)):
    ax = plt.subplot(2, 3, idx + 1)
    ax.set_facecolor(COLORS['background'])

    data = eval_df[['Algorithm', metric]].dropna()
    bars = ax.bar(data['Algorithm'], data[metric], color=color, alpha=0.8,
                  edgecolor='white', linewidth=2.5)

    # Add gradient effect
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Blues(0.5 + 0.5 * (i / len(bars))))

    ax.set_xlabel('Algorithm', fontweight='semibold')
    ax.set_ylabel(metric, fontweight='semibold')
    ax.set_title(f'{metric} Comparison', fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='semibold')

plt.tight_layout()
plt.show()
print("Displayed: Metrics comparison")

# CLUSTER PROFILING
print("\n[STEP 5] Cluster Profiling")
print("-" * 80)

print(f"Analyzing {best_algorithm} clusters...")
best_labels = clustering_results[best_algorithm]

df_profile = X_features.copy()
df_profile['Cluster'] = best_labels
cluster_profiles = df_profile.groupby('Cluster').mean()

global_mean = X_features.mean()
global_std = X_features.std().replace(0, 1e-6)

print("\nCluster Characteristics:")
print("-" * 80)

for cluster in sorted(cluster_profiles.index):
    if cluster == -1:
        print(f"\nCluster: NOISE (outliers)")
    else:
        print(f"\nCluster {cluster}:")

    n_samples = (best_labels == cluster).sum()
    pct = n_samples / len(best_labels) * 100
    print(f"  Size: {n_samples} samples ({pct:.1f}%)")

    cluster_means = cluster_profiles.loc[cluster]
    z_scores = (cluster_means - global_mean) / global_std
    important = z_scores[abs(z_scores) > 0.5].sort_values(ascending=False).head(5)

    if len(important) > 0:
        print("  Top distinguishing features:")
        for feature, z in important.items():
            direction = "Higher" if z > 0 else "Lower"
            print(f"    • {feature}: {direction} than average (Z={z:+.2f})")
    else:
        print("  No strongly distinguishing features")

cluster_profiles.to_csv(f'{OUTPUT_DIR}/cluster_profiles.csv')
print(f"\nSaved: {OUTPUT_DIR}/cluster_profiles.csv")

print("\n" + "-" * 80)
print("Cluster vs Ground Truth Alignment:")
print("-" * 80)

alignment = pd.crosstab(best_labels, y_true, margins=True)
print(alignment)
alignment.to_csv(f'{OUTPUT_DIR}/cluster_alignment.csv')
print(f"Saved: {OUTPUT_DIR}/cluster_alignment.csv")

# VISUALIZATION
print("\n[STEP 6] Creating Visualizations")
print("-" * 80)

print("Applying t-SNE for 2D projection (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca)
print("Done!")

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_pca)
print(f"2D PCA variance: {pca_2d.explained_variance_ratio_.sum() * 100:.2f}%")

# Enhanced t-SNE visualization for all algorithms
n_algos = len(clustering_results)
n_cols = (n_algos + 1 + 1) // 2
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('white')
fig.suptitle('Clustering Results (t-SNE Visualization)', fontsize=18, fontweight='bold', y=0.995)

# Plot ground truth
ax = plt.subplot(2, n_cols, 1)
ax.set_facecolor(COLORS['background'])
for i, label in enumerate(sorted(y_true.unique())):
    mask = y_true == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               label=label, alpha=0.7, s=50,
               color=COLORS['palette'][i % len(COLORS['palette'])],
               edgecolors='white', linewidths=0.5)
ax.set_title('Ground Truth', fontweight='bold', pad=12, fontsize=13)
ax.set_xlabel('t-SNE Dimension 1', fontweight='semibold')
ax.set_ylabel('t-SNE Dimension 2', fontweight='semibold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.2, linestyle='--')

# Plot each clustering algorithm
for idx, (algo_name, labels) in enumerate(clustering_results.items(), start=2):
    ax = plt.subplot(2, n_cols, idx)
    ax.set_facecolor(COLORS['background'])

    unique_labels = np.unique(labels)
    colors_algo = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=COLORS['noise'], label='Noise', alpha=0.4, s=30,
                       marker='x', linewidths=1.5)
        else:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[colors_algo[i]], label=f'C{label}', alpha=0.7, s=50,
                       edgecolors='white', linewidths=0.5)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    title = f'{algo_name}\n({n_clusters} clusters'
    if n_noise > 0:
        title += f', {n_noise} noise)'
    else:
        title += ')'

    ax.set_title(title, fontweight='bold', pad=12, fontsize=13)
    ax.set_xlabel('t-SNE Dimension 1', fontweight='semibold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='semibold')
    ax.legend(fontsize=9, ncol=2, frameon=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.show()
print("Displayed: t-SNE visualization for all algorithms")

# Best model visualization (enhanced)
fig = plt.figure(figsize=(12, 9))
fig.patch.set_facecolor('white')
ax = plt.gca()
ax.set_facecolor(COLORS['background'])

best_labels_vis = clustering_results[best_algorithm]
unique_labels = np.unique(best_labels_vis)
colors_best = plt.cm.viridis(np.linspace(0.15, 0.95, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = best_labels_vis == label
    if label == -1:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=COLORS['noise'], label='Noise', alpha=0.4, s=40,
                   marker='x', linewidths=2, zorder=1)
    else:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors_best[i]], label=f'Cluster {label}', alpha=0.75, s=70,
                   edgecolors='white', linewidths=1.5, zorder=2)

ax.set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='semibold')
ax.set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='semibold')
ax.set_title(f'Best Clustering Result: {best_algorithm}', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, frameon=True, shadow=True, loc='best')
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.show()
print("Displayed: Best model t-SNE visualization")

# FINAL SUMMARY
print("\n" + "=" * 80)
print("CLUSTERING ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nDataset: {len(X_features)} samples, {X_features.shape[1]} features")
print(f"PCA reduced to: {X_pca.shape[1]} dimensions")
print(f"Optimal k: {optimal_k}")
print(f"Best algorithm: {best_algorithm}")

print("\nAlgorithm Rankings (by Silhouette Score):")
ranked = eval_df.sort_values('Silhouette', ascending=False)
for i, row in ranked.iterrows():
    print(f"  {row['Algorithm']:15s}: {row['Silhouette']:.4f}")

print("\nInterpretation:")
print("  • Higher Silhouette Score = Better cluster separation")
print(f"  • {best_algorithm} achieved the best clustering quality")
print("  • Check cluster profiles to understand what each cluster represents")

print("\n" + "=" * 80)
print("All outputs saved to:", OUTPUT_DIR)
print("=" * 80)