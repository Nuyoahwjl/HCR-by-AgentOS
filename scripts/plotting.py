"""
Publication-quality figures for the HCR project.
Style: clean academic / paper-ready, hatched bars, dashed gridlines, bold labels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

BG   = 'white'
GRID = dict(axis='y', color='#BBBBBB', linestyle='--', linewidth=0.8, alpha=0.7)

# Colour + hatch palette (8 distinct combos - brighter, more modern colors)
PALETTE = [
    ('#2196F3', '////'),   # vibrant blue
    ('#4CAF50', '\\\\\\\\'), # vibrant green
    ('#F44336', 'xxxx'),   # vibrant red
    ('#FF9800', '----'),   # vibrant orange
    ('#9C27B0', '||||'),   # vibrant purple
    ('#00BCD4', '++++'),   # vibrant cyan
    ('#E91E63', '....'),   # vibrant pink
    ('#8BC34A', 'oooo'),   # vibrant light green
]

def bar_style(ax, colors_hatches, n_groups, n_bars, x, width, values_list,
              labels, ylabel, title, ylim=None, fmt='{:.2f}', rotation=0):
    """Draw a grouped bar chart with hatching, labels and dashed grid."""
    ax.set_facecolor(BG)
    ax.grid(**GRID)
    ax.set_axisbelow(True)

    for i, (vals, label) in enumerate(zip(values_list, labels)):
        c, h = colors_hatches[i]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=c, hatch=h, edgecolor='black', linewidth=0.8,
                      alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim[1] - ylim[0]) * 0.012 if ylim else bar.get_height() * 1.02,
                    fmt.format(v),
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels if n_groups == 1 else None, rotation=rotation, ha='right' if rotation else 'center')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════
ablation_configs = ['A: Baseline', 'B: Hybrid RAG', 'C: Multi-Agent',
                    'D: Query Decomp', 'E: Reflection', 'F: Full System']
ablation_metrics = {
    'Accuracy':  [0.45, 0.52, 0.58, 0.55, 0.60, 0.62],
    'Coverage':  [0.50, 0.58, 0.62, 0.60, 0.65, 0.68],
    'Diversity': [0.60, 0.62, 0.68, 0.65, 0.70, 0.72],
    'Precision': [0.40, 0.48, 0.52, 0.50, 0.55, 0.58],
}
ablation_latency = [36.4, 40.5, 140.3, 47.7, 143.2, 140.0]

retrieval_methods = ['Pure Vector', 'BM25 Only', 'Hybrid (RRF)', 'Hybrid+Reranker']
recall_k   = {'Recall@1': [0.35,0.42,0.55,0.58],
               'Recall@3': [0.52,0.58,0.68,0.72],
               'Recall@5': [0.62,0.65,0.75,0.78]}
mrr_ndcg   = {'MRR':     [0.45,0.52,0.65,0.68],
               'nDCG@5': [0.48,0.55,0.67,0.70]}

latency_components = ['Query Decomposition','Hybrid Retrieval','Reranking',
                      'Symptom Analysis','Risk Assessment','Recommendation',
                      'Safety Check','Synthesis']
latency_times = [2.1, 15.3, 8.7, 32.5, 28.4, 45.2, 22.8, 15.0]

agents    = ['SymptomAnalyzer','RiskAssessor','RecommendationAgent','SafetyChecker']
agent_avg = [32.5, 28.4, 45.2, 22.8]
agent_std = [5.2,   4.8,   6.1,  3.9]

safety_metrics = ['Hallucination Rate','Omission Rate','Contraindication\nDetection']
safety_base    = [0.91, 0.97, 0.65]
safety_full    = [0.28, 0.32, 0.85]

facets       = ['Symptom','History','Demographic','Risk Combination','Full Profile']
facet_pct    = [35, 25, 15, 20, 5]

syn_conds    = ['Without Synonyms','With Synonyms']
syn_recall   = [0.68, 0.78]
syn_ndcg     = [0.62, 0.70]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Ablation Study Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════
def fig1_ablation():
    metrics = list(ablation_metrics.keys())
    n_configs = len(ablation_configs)
    n_metrics = len(metrics)
    x = np.arange(n_configs)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(**GRID)
    ax.set_axisbelow(True)

    for i, metric in enumerate(metrics):
        c, h = PALETTE[i]
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, ablation_metrics[metric], width,
                      label=metric, color=c, hatch=h,
                      edgecolor='black', linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, ablation_metrics[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_configs, rotation=20, ha='right', fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim(0.30, 0.82)
    ax.set_xlabel('Configuration')
    ax.set_title('Ablation Study: Impact of System Components')
    ax.legend(loc='upper left', ncol=2)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_ablation_study.png')
    plt.close()
    print('Saved fig1_ablation_study.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Retrieval Comparison (line + bar side-by-side)
# ═══════════════════════════════════════════════════════════════════════
def fig2_retrieval():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    k_vals = [1, 3, 5]
    line_colors = [p[0] for p in PALETTE[:4]]  # 使用PALETTE前4种颜色
    markers = ['o','s','^','D']

    # — left: Recall@K lines —
    ax1.set_facecolor(BG)
    ax1.grid(**GRID)
    ax1.set_axisbelow(True)
    for j, method in enumerate(retrieval_methods):
        recalls = [recall_k[f'Recall@{k}'][j] for k in k_vals]
        ax1.plot(k_vals, recalls, marker=markers[j], color=line_colors[j],
                 linewidth=2.2, markersize=8, label=method)
        for k, r in zip(k_vals, recalls):
            ax1.text(k, r + 0.012, f'{r:.2f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=line_colors[j])
    ax1.set_xticks(k_vals)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Recall@K')
    ax1.set_title('Recall@K Comparison')
    ax1.set_ylim(0.28, 0.88)
    ax1.legend(fontsize=10)

    # — right: MRR & nDCG@5 bars —
    ax2.set_facecolor(BG)
    ax2.grid(**GRID)
    ax2.set_axisbelow(True)
    x = np.arange(len(retrieval_methods))
    width = 0.32
    for i, (metric, vals) in enumerate(mrr_ndcg.items()):
        c, h = PALETTE[i]
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, vals, width, label=metric,
                       color=c, hatch=h, edgecolor='black', linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(retrieval_methods, rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('Score')
    ax2.set_title('MRR and nDCG@5 Comparison')
    ax2.set_ylim(0.35, 0.80)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_retrieval_comparison.png')
    plt.close()
    print('Saved fig2_retrieval_comparison.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Latency Waterfall (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════
def fig3_latency():
    order   = np.argsort(latency_times)
    comps   = [latency_components[i] for i in order]
    times   = [latency_times[i]      for i in order]

    # Cycle through PALETTE for visual variety
    cp = [PALETTE[i % len(PALETTE)] for i in range(len(comps))]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(**{**GRID, 'axis': 'x'})
    ax.set_axisbelow(True)

    y = np.arange(len(comps))
    for yi, (t, (c, h)) in enumerate(zip(times, cp)):
        ax.barh(yi, t, color=c, hatch=h, edgecolor='black', linewidth=0.8, height=0.6, alpha=0.88)
        ax.text(t + 0.6, yi, f'{t:.1f}s', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(comps, fontsize=11)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Component Latency Breakdown')
    total = sum(latency_times)
    ax.text(0.97, 0.04, f'Total: {total:.1f} s',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF2CC',
                      edgecolor='#DAA520', linewidth=1.2))
    ax.set_xlim(0, max(times) * 1.18)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_latency_waterfall.png')
    plt.close()
    print('Saved fig3_latency_waterfall.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Agent Response Times with Error Bars
# ═══════════════════════════════════════════════════════════════════════
def fig4_agents():
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(**GRID)
    ax.set_axisbelow(True)

    x = np.arange(len(agents))
    for xi, (avg, std) in enumerate(zip(agent_avg, agent_std)):
        c, h = PALETTE[xi]
        bar = ax.bar(xi, avg, color=c, hatch=h, edgecolor='black',
                     linewidth=0.8, alpha=0.88, width=0.5)
        ax.errorbar(xi, avg, yerr=std, fmt='none', ecolor='#333333',
                    elinewidth=2, capsize=6, capthick=2)
        ax.text(xi, avg + std + 1.2, f'{avg:.1f}±{std:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Average Response Time by Agent')
    ax.set_ylim(0, max(agent_avg) + max(agent_std) + 12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_agent_response_times.png')
    plt.close()
    print('Saved fig4_agent_response_times.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Safety Metrics Comparison
# ═══════════════════════════════════════════════════════════════════════
def fig5_safety():
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(**GRID)
    ax.set_axisbelow(True)

    x = np.arange(len(safety_metrics))
    width = 0.32
    for i, (label, vals, (c, h)) in enumerate(zip(
            ['Baseline', 'Full System'],
            [safety_base, safety_full],
            [PALETTE[2], PALETTE[1]])):  # 使用PALETTE的红色和绿色
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=c, hatch=h, edgecolor='black', linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(safety_metrics, fontsize=12)
    ax.set_ylabel('Score')
    ax.set_title('Safety Metrics: Baseline vs. Full System')
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_safety_metrics.png')
    plt.close()
    print('Saved fig5_safety_metrics.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Query Facet Contribution (donut + bar)
# ═══════════════════════════════════════════════════════════════════════
def fig6_facets():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)

    colors_p = [p[0] for p in PALETTE[:len(facets)]]
    hatches_p = [p[1] for p in PALETTE[:len(facets)]]

    # — donut —
    wedges, texts, autotexts = ax1.pie(
        facet_pct, labels=facets, autopct='%1.1f%%',
        colors=colors_p, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    # draw a white circle to make it a donut
    centre_circle = plt.Circle((0, 0), 0.45, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Query Facet Contribution', pad=12)

    # — bar with hatches —
    ax2.set_facecolor(BG)
    ax2.grid(**GRID)
    ax2.set_axisbelow(True)
    x = np.arange(len(facets))
    for xi, (val, c, h) in enumerate(zip(facet_pct, colors_p, hatches_p)):
        ax2.bar(xi, val, color=c, hatch=h, edgecolor='black',
                linewidth=0.8, alpha=0.88, width=0.55)
        ax2.text(xi, val + 0.5, f'{val}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(facets, rotation=20, ha='right', fontsize=11)
    ax2.set_ylabel('Contribution (%)')
    ax2.set_title('Query Facet Contribution (Bar)', pad=12)
    ax2.set_ylim(0, 42)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_query_facets.png')
    plt.close()
    print('Saved fig6_query_facets.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Synonym Expansion Effect
# ═══════════════════════════════════════════════════════════════════════
def fig7_synonym():
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(**GRID)
    ax.set_axisbelow(True)

    x = np.arange(len(syn_conds))
    width = 0.32
    datasets = [('Recall@5', syn_recall, PALETTE[0]),
                ('nDCG@5',   syn_ndcg,   PALETTE[1])]
    for i, (label, vals, (c, h)) in enumerate(datasets):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=c, hatch=h, edgecolor='black', linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # improvement annotations
    r_imp = (syn_recall[1] - syn_recall[0]) / syn_recall[0] * 100
    n_imp = (syn_ndcg[1]   - syn_ndcg[0])   / syn_ndcg[0]   * 100
    ax.text(0.97, 0.96,
            f'Recall@5: +{r_imp:.1f}%\nnDCG@5: +{n_imp:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                      edgecolor='#1565C0', linewidth=1.2))  # 使用蓝色系

    ax.set_xticks(x)
    ax.set_xticklabels(syn_conds, fontsize=12)
    ax.set_ylabel('Score')
    ax.set_title('Effect of Medical Synonym Expansion')
    ax.set_ylim(0.55, 0.87)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_synonym_effect.png')
    plt.close()
    print('Saved fig7_synonym_effect.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8 — System Architecture Flow Chart (青蓝色系)
# ═══════════════════════════════════════════════════════════════════════
def fig8_architecture():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # 青蓝色系配色
    cols = {
        'input':   ('#E0F7FA', '#00838F'),  # 青蓝色
        'process': ('#E8F5E9', '#2E7D32'),  # 绿色
        'agent':   ('#FFF3E0', '#EF6C00'),  # 橙色
        'output':  ('#F3E5F5', '#7B1FA2'),  # 紫色
    }

    def box(x, y, w, h, text, kind, fs=9):
        fc, ec = cols[kind]
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0.15',
                               facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', wrap=True)

    def arrow(xy, xytext, rad=0.0, color='#2C3E50', lw=1.5):
        ax.annotate('', xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, connectionstyle=f'arc3,rad={rad}'))

    # Nodes
    box(0.3, 8.5, 2.0, 1.0, 'User Profile\n(Gender, Age,\nHistory, Symptoms)', 'input', fs=8)
    box(3.2, 8.5, 2.0, 1.0, 'Query\nDecomposer', 'process')
    box(6.1, 8.5, 2.0, 1.0, 'Hybrid\nRetrieval', 'process')
    agents_pos = [(0.5, 6.2), (2.8, 6.2), (5.1, 6.2), (7.4, 6.2)]
    agent_labels = ['Symptom\nAnalyzer', 'Risk\nAssessor', 'Recommendation\nAgent', 'Safety\nChecker']
    for (ax_, ay_), lbl in zip(agents_pos, agent_labels):
        box(ax_, ay_, 1.8, 1.0, lbl, 'agent', fs=9)
    box(2.8, 4.2, 4.5, 1.0, 'Coordinator  (Reflection Loop)', 'process', fs=10)
    box(3.2, 2.2, 3.6, 1.0, 'Final Recommendation\nwith Citations', 'output', fs=10)

    # Arrows
    arrow((3.2, 9.0), (2.3, 9.0))
    arrow((6.1, 9.0), (5.2, 9.0))
    for (ax_, ay_) in agents_pos:
        arrow((ax_+0.9, 7.2), (7.1, 8.7), rad=-0.3)
    for (ax_, ay_) in agents_pos:
        arrow((5.05, 4.9), (ax_+0.9, 6.2), rad=0.1)
    arrow((5.05, 3.2), (5.05, 4.2), lw=2)
    # Reflection loop
    # ax.annotate('', xy=(7.6, 4.5), xytext=(7.6, 5.2),
    #             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8,
    #                            connectionstyle='arc3,rad=0.4'))
    # ax.text(8.0, 4.85, 'Revision', fontsize=9, color='#C0392B', fontweight='bold')

    ax.set_title('Multi-Agent RAG System Architecture',
                 fontsize=17, fontweight='bold', pad=16, y=0.98)

    legend_handles = [mpatches.Patch(facecolor=v[0], edgecolor=v[1], label=k.capitalize(), linewidth=1.5)
                      for k, v in cols.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig8_system_architecture.png')
    plt.close()
    print('Saved fig8_system_architecture.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 9 — Retrieval Pipeline Flow Chart (蓝绿渐变色系)
# ═══════════════════════════════════════════════════════════════════════
def fig9_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')

    # 使用PALETTE颜色风格
    cols = {
        'query':     ('#E3F2FD', PALETTE[0][0]),  # 使用PALETTE蓝色
        'retrieval': ('#E8F5E8', PALETTE[1][0]),  # 使用PALETTE绿色
        'fusion':    ('#FFF8E1', PALETTE[3][0]),  # 使用PALETTE橙色
        'rerank':    ('#FFEBEE', PALETTE[2][0]),  # 使用PALETTE红色
    }

    def box(x, y, w, h, text, kind, fs=10):
        fc, ec = cols[kind]
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0.12',
                               facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold')

    def arrow(xy, xytext, rad=0.0):
        ax.annotate('', xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle='->', color='#2C3E50',
                                   lw=1.6, connectionstyle=f'arc3,rad={rad}'))

    box(0.3, 3.8, 2.0, 1.5, 'User Query', 'query')
    box(3.0, 3.8, 2.2, 1.5, 'Query\nDecomposition', 'query')
    box(6.2, 4.6, 2.1, 1.0, 'BM25\nRetrieval',       'retrieval', fs=9)
    box(6.2, 3.0, 2.1, 1.0, 'Dense Vector\nRetrieval','retrieval', fs=9)
    box(9.4, 3.8, 2.2, 1.5, 'Reciprocal\nRank Fusion','fusion')
    box(9.4, 1.8, 2.2, 1.5, 'Cross-Encoder\nReranking','rerank')
    box(9.4, 0.2, 2.2, 1.0, 'Top-K Results',          'rerank')

    arrow((3.0, 4.55), (2.3, 4.55))
    arrow((6.2, 5.1),  (5.2, 5.0),  rad=-0.3)
    arrow((6.2, 3.5),  (5.2, 4.1),  rad=0.3)
    arrow((9.4, 5.0),  (8.3, 5.1),  rad=0.2)
    arrow((9.4, 4.2),  (8.3, 3.5),  rad=-0.2)
    arrow((10.5, 3.3), (10.5, 3.8))
    arrow((10.5, 1.2), (10.5, 1.8))

    ax.set_title('Hybrid Retrieval Pipeline with Reranking',
                 fontsize=16, fontweight='bold', pad=14)
    legend_handles = [mpatches.Patch(facecolor=v[0], edgecolor=v[1],
                                     label={'query':'Query Processing','retrieval':'Retrieval',
                                            'fusion':'Fusion','rerank':'Reranking'}[k],
                                     linewidth=1.5)
                      for k, v in cols.items()]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig9_retrieval_pipeline.png')
    plt.close()
    print('Saved fig9_retrieval_pipeline.png')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 10 — Evaluation Radar Chart (蓝红对比色系)
# ═══════════════════════════════════════════════════════════════════════
def fig10_radar():
    categories = ['Accuracy', 'Coverage', 'Diversity', 'Precision', 'Speed']
    N = len(categories)

    baseline_speed = 1 / ablation_latency[0] * 100
    full_speed     = 1 / ablation_latency[5] * 100

    def normalize_speed(s, ref=baseline_speed):
        return min(s / ref, 1.0)

    baseline_vals = [0.45, 0.50, 0.60, 0.40, normalize_speed(baseline_speed)]
    full_vals     = [0.62, 0.68, 0.72, 0.58, normalize_speed(full_speed)]

    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    baseline_vals += baseline_vals[:1]
    full_vals     += full_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG,
                           subplot_kw=dict(projection='polar'))
    ax.set_facecolor('#FAFAFA')

    # 使用PALETTE颜色风格
    ax.plot(angles, baseline_vals, 'o-', linewidth=2.2, color=PALETTE[0][0], label='Baseline (A)')
    ax.fill(angles, baseline_vals, alpha=0.15, color=PALETTE[0][0])
    ax.plot(angles, full_vals, 's-', linewidth=2.2, color=PALETTE[2][0], label='Full System (F)')
    ax.fill(angles, full_vals, alpha=0.15, color=PALETTE[2][0])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=9)
    ax.yaxis.grid(color='#CCCCCC', linestyle='--', linewidth=0.8)
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.8)
    ax.set_title('System Performance Comparison\n(Radar Chart)',
                 fontsize=15, fontweight='bold', pad=28)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=12)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig10_evaluation_radar.png')
    plt.close()
    print('Saved fig10_evaluation_radar.png')


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating all figures...\n')
    fig1_ablation()
    fig2_retrieval()
    fig3_latency()
    fig4_agents()
    fig5_safety()
    fig6_facets()
    fig7_synonym()
    fig8_architecture()
    fig9_pipeline()
    fig10_radar()
    print('\nAll done!')