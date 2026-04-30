"""
=============================================================================
  AI-Driven Customer Segmentation System — Streamlit App
  Dataset: Mall Customer Segmentation (Kaggle)
  Techniques: K-Means Clustering + PCA + Feature Scaling
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import io
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0F0F1A; color: white; }
    .stApp { background-color: #0F0F1A; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #4ECDC4 !important; }
    .stMetric { background: #1A1A2E; border-radius: 8px; padding: 10px; }
    .stDataFrame { background: #1A1A2E; }
    .metric-card {
        background: #1A1A2E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .segment-badge {
        display: inline-block;
        background: #2A2A4E;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85em;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1. GENERATE DATASET
# ─────────────────────────────────────────────
@st.cache_data
def generate_mall_customers(n=200, seed=42):
    """Synthetic Mall Customers dataset matching Kaggle schema."""
    np.random.seed(seed)
    data = {
        'CustomerID': range(1, n + 1),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.44, 0.56]),
    }
    ages = np.concatenate([
        np.random.normal(25, 4, int(n * 0.25)),
        np.random.normal(35, 5, int(n * 0.30)),
        np.random.normal(45, 6, int(n * 0.25)),
        np.random.normal(60, 7, int(n * 0.20)),
    ])
    np.random.shuffle(ages)
    data['Age'] = np.clip(ages, 18, 70).astype(int)[:n]

    incomes, scores = [], []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.20:
            incomes.append(np.random.normal(85, 8))
            scores.append(np.random.normal(20, 8))
        elif r < 0.40:
            incomes.append(np.random.normal(80, 8))
            scores.append(np.random.normal(82, 8))
        elif r < 0.60:
            incomes.append(np.random.normal(25, 6))
            scores.append(np.random.normal(80, 8))
        elif r < 0.80:
            incomes.append(np.random.normal(25, 6))
            scores.append(np.random.normal(20, 8))
        else:
            incomes.append(np.random.normal(55, 10))
            scores.append(np.random.normal(50, 12))

    data['Annual_Income_k'] = np.clip(incomes, 15, 137).astype(int)[:n]
    data['Spending_Score'] = np.clip(scores, 1, 99).astype(int)[:n]

    freq_base = (np.array(data['Spending_Score']) / 20 + np.array(data['Annual_Income_k']) / 40)
    data['Purchase_Frequency'] = np.clip(
        (freq_base + np.random.normal(0, 1, n)).astype(int), 1, 15)

    categories = ['Electronics', 'Fashion', 'Groceries', 'Sports', 'Home & Decor']
    data['Preferred_Category'] = np.random.choice(categories, n)

    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING & CLUSTERING
# ─────────────────────────────────────────────
@st.cache_data
def run_pipeline(n_customers, seed):
    df = generate_mall_customers(n_customers, seed)

    df['Gender_Encoded'] = (df['Gender'] == 'Female').astype(int)
    cat_map = {c: i for i, c in enumerate(df['Preferred_Category'].unique())}
    df['Category_Encoded'] = df['Preferred_Category'].map(cat_map)

    features = ['Age', 'Annual_Income_k', 'Spending_Score',
                'Purchase_Frequency', 'Gender_Encoded', 'Category_Encoded']
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optimal K
    inertia, sil_scores = [], []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    optimal_k = list(K_range)[np.argmax(sil_scores)]

    # K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca2 = PCA(n_components=2, random_state=42)
    X_pca2 = pca2.fit_transform(X_scaled)
    df['PC1'], df['PC2'] = X_pca2[:, 0], X_pca2[:, 1]

    pca3 = PCA(n_components=3, random_state=42)
    X_pca3 = pca3.fit_transform(X_scaled)
    df['PC3'] = X_pca3[:, 2]

    var2 = pca2.explained_variance_ratio_
    var3 = pca3.explained_variance_ratio_

    # Cluster Profile
    cluster_profile = df.groupby('Cluster')[
        ['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency']
    ].mean().round(1)
    cluster_profile['Size'] = df['Cluster'].value_counts().sort_index()
    cluster_profile['% Share'] = (cluster_profile['Size'] / len(df) * 100).round(1)

    def label_cluster(row):
        income = row['Annual_Income_k']
        spend = row['Spending_Score']
        if income > 65 and spend > 65:
            return "💎 Champions"
        elif income > 65 and spend < 35:
            return "💰 High Income, Low Spend"
        elif income < 40 and spend > 65:
            return "🛍️ Budget Big Spenders"
        elif income < 40 and spend < 35:
            return "💤 Low Priority"
        else:
            return "🎯 Standard Customers"

    cluster_profile['Segment'] = cluster_profile.apply(label_cluster, axis=1)

    return (df, cluster_profile, kmeans, pca2, pca3,
            var2, var3, list(K_range), inertia, sil_scores, optimal_k)


# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────
PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
           '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#82E0AA']

DARK_BG = '#0F0F1A'
PLOT_BG = '#1A1A2E'


# ─────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    buf.seek(0)
    return buf


def dark_fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(DARK_BG)
    if hasattr(axes, '__iter__'):
        for ax in np.array(axes).flat:
            ax.set_facecolor(PLOT_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
    else:
        axes.set_facecolor(PLOT_BG)
        for spine in axes.spines.values():
            spine.set_edgecolor('#333')
    return fig, axes


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    n_customers = st.slider("Number of Customers", 100, 500, 200, 50)
    seed = st.number_input("Random Seed", 0, 999, 42)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **Algorithm**: K-Means Clustering  
    **Dimensionality Reduction**: PCA  
    **Scaling**: StandardScaler  
    **Dataset**: Synthetic Mall Customers
    """)

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
with st.spinner("Running segmentation pipeline..."):
    (df, cluster_profile, kmeans, pca2, pca3,
     var2, var3, K_range, inertia, sil_scores, optimal_k) = run_pipeline(n_customers, seed)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🛍️ AI-Driven Customer Segmentation")
st.markdown("##### K-Means Clustering · PCA · Feature Scaling · Business Intelligence")
st.divider()

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 Total Customers", len(df))
col2.metric("🔢 Optimal Clusters", optimal_k)
col3.metric("📐 Silhouette Score", f"{max(sil_scores):.3f}")
col4.metric("📉 PCA 2D Variance", f"{var2.sum()*100:.1f}%")
col5.metric("📉 PCA 3D Variance", f"{var3.sum()*100:.1f}%")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Optimal K", "🗺️ PCA Clusters", "🔥 Heatmap",
    "📈 Scatter Analysis", "🕸️ Radar Chart", "📋 Report"
])

# ── TAB 1: Elbow & Silhouette ──────────────────
with tab1:
    st.subheader("Optimal Cluster Selection")
    fig, axes = dark_fig(1, 2, figsize=(13, 5))

    axes[0].plot(K_range, inertia, 'o-', color='#4ECDC4', lw=2.5, ms=8)
    axes[0].axvline(optimal_k, color='#FF6B6B', ls='--', lw=1.5, label=f'Optimal K={optimal_k}')
    axes[0].set_title('Elbow Method — Inertia vs K', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Number of Clusters (K)', color='#aaa')
    axes[0].set_ylabel('Inertia', color='#aaa')
    axes[0].tick_params(colors='#aaa')
    axes[0].legend(facecolor=PLOT_BG, labelcolor='white')

    axes[1].plot(K_range, sil_scores, 's-', color='#FF6B6B', lw=2.5, ms=8)
    axes[1].axvline(optimal_k, color='#4ECDC4', ls='--', lw=1.5, label=f'Best K={optimal_k}')
    axes[1].set_title('Silhouette Score vs K', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Number of Clusters (K)', color='#aaa')
    axes[1].set_ylabel('Silhouette Score', color='#aaa')
    axes[1].tick_params(colors='#aaa')
    axes[1].legend(facecolor=PLOT_BG, labelcolor='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2: PCA Scatter ────────────────────────
with tab2:
    st.subheader("Customer Segments in PCA Space")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**2D PCA Projection**")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PLOT_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

        for c in sorted(df['Cluster'].unique()):
            mask = df['Cluster'] == c
            seg = cluster_profile.loc[c, 'Segment']
            ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                       c=PALETTE[c], s=60, alpha=0.85, edgecolors='white', lw=0.3,
                       label=f'C{c}: {seg}')

        centres_pca = pca2.transform(kmeans.cluster_centers_)
        ax.scatter(centres_pca[:, 0], centres_pca[:, 1],
                   c='white', s=180, marker='*', zorder=5, label='Centroids')

        ax.set_title(f'PCA 2D  ({var2.sum()*100:.1f}% variance)', color='white', fontsize=12)
        ax.set_xlabel(f'PC1 ({var2[0]*100:.1f}%)', color='#aaa')
        ax.set_ylabel(f'PC2 ({var2[1]*100:.1f}%)', color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.legend(facecolor='#16213E', labelcolor='white', fontsize=7.5)
        ax.grid(True, alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Cluster Size Distribution**")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)

        labels_pie = [f'C{i}\n{cluster_profile.loc[i, "Segment"]}\n({cluster_profile.loc[i, "% Share"]}%)'
                      for i in cluster_profile.index]
        sizes = cluster_profile['Size'].values
        colors = [PALETTE[i] for i in cluster_profile.index]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors, autopct='%1.0f%%',
            startangle=140, pctdistance=0.75,
            wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
        for t in texts:
            t.set_color('white')
            t.set_fontsize(7.5)
        for a in autotexts:
            a.set_color('black')
            a.set_fontweight('bold')
            a.set_fontsize(9)
        ax.set_title('Cluster Size Distribution', color='white', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 3: Heatmap ────────────────────────────
with tab3:
    st.subheader("Cluster Feature Heatmap")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PLOT_BG)

    heat_data = cluster_profile[['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency']].copy()
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

    sns.heatmap(heat_norm.T, annot=heat_data.T.values, fmt='.1f',
                cmap='YlOrRd', ax=ax,
                xticklabels=[f'C{i}: {cluster_profile.loc[i, "Segment"]}' for i in cluster_profile.index],
                yticklabels=['Age', 'Annual Income (k$)', 'Spending Score', 'Purchase Freq'],
                linewidths=0.5, linecolor='#333',
                annot_kws={'size': 11, 'color': 'black', 'weight': 'bold'},
                cbar_kws={'label': 'Normalised Value'})

    ax.set_title('Cluster Feature Heatmap', color='white', fontsize=14, pad=12)
    ax.tick_params(axis='x', colors='white', labelsize=8.5, rotation=20)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    ax.figure.axes[-1].tick_params(colors='white')
    ax.figure.axes[-1].yaxis.label.set_color('white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 4: Scatter Analysis ───────────────────
with tab4:
    st.subheader("Customer Distribution Analysis")
    fig, axes = dark_fig(1, 2, figsize=(14, 5))

    for c in sorted(df['Cluster'].unique()):
        mask = df['Cluster'] == c
        seg = cluster_profile.loc[c, 'Segment']
        axes[0].scatter(df.loc[mask, 'Annual_Income_k'], df.loc[mask, 'Spending_Score'],
                        c=PALETTE[c], s=60, alpha=0.85, edgecolors='white', lw=0.3,
                        label=f'C{c}: {seg}')

    axes[0].set_title('Annual Income vs Spending Score', color='white', fontsize=12)
    axes[0].set_xlabel('Annual Income (k$)', color='#aaa')
    axes[0].set_ylabel('Spending Score (1–99)', color='#aaa')
    axes[0].tick_params(colors='#aaa')
    axes[0].legend(facecolor='#16213E', labelcolor='white', fontsize=7.5)
    axes[0].grid(True, alpha=0.1, color='white')

    for c in sorted(df['Cluster'].unique()):
        mask = df['Cluster'] == c
        seg = cluster_profile.loc[c, 'Segment']
        axes[1].scatter(df.loc[mask, 'Age'], df.loc[mask, 'Spending_Score'],
                        c=PALETTE[c], s=60, alpha=0.85, edgecolors='white', lw=0.3,
                        label=f'C{c}: {seg}')

    axes[1].set_title('Age vs Spending Score', color='white', fontsize=12)
    axes[1].set_xlabel('Age', color='#aaa')
    axes[1].set_ylabel('Spending Score (1–99)', color='#aaa')
    axes[1].tick_params(colors='#aaa')
    axes[1].legend(facecolor='#16213E', labelcolor='white', fontsize=7.5)
    axes[1].grid(True, alpha=0.1, color='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Gender distribution
    st.markdown("**Gender Distribution per Cluster**")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    gender_cluster = df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
    gender_cluster.plot(kind='bar', ax=ax,
                        color=['#FF6B9D', '#45B7D1'], edgecolor=DARK_BG, width=0.6)
    ax.set_title('Gender Distribution per Cluster', color='white', fontsize=12)
    ax.set_xlabel('Cluster', color='#aaa')
    ax.set_ylabel('Count', color='#aaa')
    ax.tick_params(colors='white', rotation=0)
    ax.legend(facecolor='#16213E', labelcolor='white')
    ax.grid(axis='y', alpha=0.15, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 5: Radar Chart ────────────────────────
with tab5:
    st.subheader("Cluster Radar Chart — Feature Comparison")
    metrics = ['Age', 'Annual_Income_k', 'Spending_Score', 'Purchase_Frequency']
    metric_labels = ['Age', 'Income', 'Spending\nScore', 'Purchase\nFreq']
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PLOT_BG)

    norm_profile = cluster_profile[metrics].copy()
    for col in metrics:
        norm_profile[col] = (norm_profile[col] - norm_profile[col].min()) / \
                            (norm_profile[col].max() - norm_profile[col].min() + 1e-9)

    for idx, row in norm_profile.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, 'o-', color=PALETTE[idx], lw=2.2, ms=6,
                label=f'C{idx}: {cluster_profile.loc[idx, "Segment"]}')
        ax.fill(angles, vals, color=PALETTE[idx], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, color='white', fontsize=11)
    ax.set_yticklabels([])
    ax.grid(color='#444', linewidth=0.8)
    ax.spines['polar'].set_color('#444')
    ax.set_title('Radar Chart — Feature Comparison', color='white', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#16213E', labelcolor='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 6: Report ─────────────────────────────
with tab6:
    st.subheader("📋 Cluster Profiles")
    display_df = cluster_profile.copy()
    display_df.index.name = 'Cluster'
    st.dataframe(display_df.style.background_gradient(cmap='YlOrRd', subset=['Spending_Score', 'Annual_Income_k']),
                 use_container_width=True)

    st.divider()
    st.subheader("🎯 Business Recommendations")

    recs = {
        "💎 Champions": ("High Income + High Spend",
                          "VIP loyalty programme, exclusive early access, premium offers, personal account managers. **Goal**: Retain and upsell premium product lines."),
        "💰 High Income, Low Spend": ("High earners not converting",
                                       "Targeted re-engagement via curated luxury recommendations. Understand pain points through surveys and focus groups. **Goal**: Convert savings into spend."),
        "🛍️ Budget Big Spenders": ("Low income, high engagement",
                                    "Flash sales, limited-time offers, gamification. BNPL payment options. **Goal**: Sustain high engagement despite budget constraints."),
        "💤 Low Priority": ("Low Income + Low Spend",
                             "Low-cost retention: newsletters, social media, community building. Affordable essentials and bundle deals. **Goal**: Cost-effective nurturing."),
        "🎯 Standard Customers": ("Mid-range segment",
                                   "Behavioural email triggers, ML-powered product recommendations. Cross-sell related categories, referral incentives. **Goal**: Move toward Champions."),
    }

    for segment, (desc, rec) in recs.items():
        with st.expander(f"{segment} — {desc}"):
            st.markdown(rec)

    st.divider()
    st.subheader("📊 Key Metrics Summary")
    col1, col2 = st.columns(2)
    high_val = len(df[(df['Annual_Income_k'] > 65) & (df['Spending_Score'] > 65)])
    at_risk = len(df[df['Spending_Score'] < 30])
    col1.metric("💎 High-Value Customers", high_val,
                f"{high_val/len(df)*100:.1f}% of base")
    col2.metric("⚠️ At-Risk Customers (Score < 30)", at_risk,
                f"{at_risk/len(df)*100:.1f}% of base")

    st.divider()
    # Download section
    st.subheader("⬇️ Download Data")
    csv_segments = df.to_csv(index=False).encode('utf-8')
    csv_profiles = cluster_profile.to_csv().encode('utf-8')

    col1, col2 = st.columns(2)
    col1.download_button("📥 Download Customer Segments CSV",
                         csv_segments, "customer_segments.csv", "text/csv")
    col2.download_button("📥 Download Cluster Profiles CSV",
                         csv_profiles, "cluster_profiles.csv", "text/csv")

    # Raw data preview
    with st.expander("🔎 Preview Raw Dataset"):
        st.dataframe(df.head(50), use_container_width=True)
