import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LINEAR_MODELS = ['lasso', 'ridge', 'elasticnet', 'svr']

MODEL_PRETTY = {
    'lasso': 'Lasso', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet', 'svr': 'SVR',
    'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest',
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'histgradientboosting': 'HistGradientBoosting',
}

FILL_COLOR = {'Linear': '#4A89BC', 'Non-Linear': '#E87D3E'}
EDGE_COLOR = {'Linear': '#2E6A9E', 'Non-Linear': '#C05A1F'}

MODEL_PALETTE = sns.color_palette("coolwarm", 9).as_hex()

FIG_W, FIG_H = 6, 5


def _r2(d):
    ss_res = np.sum((d['predicted_change'] - d['actual_change']) ** 2)
    ss_tot = np.sum((d['actual_change'] - d['actual_change'].mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def best_model_by_r2(cv_df):
    fold_r2 = (
        cv_df.groupby(['model', 'fold'])
        .apply(_r2, include_groups=False)
        .reset_index(name='r2')
    )
    return fold_r2.groupby('model')['r2'].median().idxmax()


def _apply_spine_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.tick_params(axis='both', labelsize=9, length=4, width=0.7)


def _two_tone_box(ax, bp, fc, ec, box_w, j):
    box_verts = bp['boxes'][0].get_path().vertices
    q1, q3 = box_verts[:, 1].min(), box_verts[:, 1].max()
    med = bp['medians'][0].get_ydata()[0]

    bp['boxes'][0].set(facecolor='none', edgecolor=ec, linewidth=0.8)
    bp['medians'][0].set(color=ec, linewidth=1.2, zorder=6)
    for w in bp['whiskers']:
        w.set(color=ec, linewidth=0.8)
    for c in bp['caps']:
        c.set(color=ec, linewidth=0.8)

    fc_rgba = to_rgba(fc)
    fc_light = tuple(min(1, ch + 0.15) for ch in fc_rgba[:3]) + (0.85,)
    fc_dark  = tuple(max(0, ch - 0.08) for ch in fc_rgba[:3]) + (0.85,)
    ax.add_patch(plt.Rectangle((j - box_w / 2, med), box_w, q3 - med,
                               facecolor=fc_light, edgecolor='none', zorder=3))
    ax.add_patch(plt.Rectangle((j - box_w / 2, q1), box_w, med - q1,
                               facecolor=fc_dark, edgecolor='none', zorder=3))
    return med


def plot_r2_by_model(cv_df, final_df=None, SELECTED_TIME=None, save=False):
    fold_r2 = (
        cv_df.groupby(['model', 'fold'])
        .apply(_r2, include_groups=False)
        .reset_index(name='r2')
    )
    fold_r2['group'] = fold_r2['model'].map(
        lambda m: 'Linear' if m in LINEAR_MODELS else 'Non-Linear'
    )
    medians = fold_r2.groupby(['model', 'group'])['r2'].median().reset_index()
    ordered = (
        medians[medians['group'] == 'Linear'].sort_values('r2', ascending=False)['model'].tolist()
        + medians[medians['group'] == 'Non-Linear'].sort_values('r2', ascending=False)['model'].tolist()
    )

    has_final = (
        final_df is not None
        and not final_df.empty
        and 'actual_change' in final_df.columns
    )
    if has_final:
        final_r2 = (
            final_df.groupby('model')
            .apply(_r2, include_groups=False)
            .rename('r2')
        )

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    box_w = 0.45

    for j, model in enumerate(ordered):
        ax.axvline(j, color='#e0e0e0', linewidth=0.5, zorder=1)
        grp = 'Linear' if model in LINEAR_MODELS else 'Non-Linear'
        fc, ec = FILL_COLOR[grp], EDGE_COLOR[grp]
        vals = fold_r2[fold_r2['model'] == model]['r2'].dropna().values

        bp = ax.boxplot(vals, positions=[j], widths=box_w,
                        patch_artist=True, manage_ticks=False,
                        flierprops=dict(marker='None'))
        _two_tone_box(ax, bp, fc, ec, box_w, j)

        np.random.seed(j)
        jit = np.random.uniform(-0.10, 0.10, len(vals))
        ax.scatter(j + jit, vals, s=6, color=ec, alpha=0.4, zorder=5)

        if has_final and model in final_r2.index:
            ax.scatter(j, final_r2[model], marker='x', s=25, color='black',
                       zorder=7, label='Final model' if j == 0 else '_', linewidth=1)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([MODEL_PRETTY.get(m, m) for m in ordered], rotation=40, ha='right')
    ax.set_ylabel('R²', labelpad=6)
    title = r"$\mathbf{R^2\ by\ model}$"
    ax.set_title(f"{title}\n(Time {SELECTED_TIME})", fontsize=12, pad=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    _apply_spine_style(ax)

    legend_handles = [
        Patch(facecolor=FILL_COLOR[g], edgecolor=EDGE_COLOR[g], linewidth=0.8,
              label=f'{g} methods')
        for g in ('Linear', 'Non-Linear')
    ]
    if has_final:
        legend_handles.append(
            Line2D([0], [0], marker='x', color='#222', linestyle='None',
                   markersize=5, label='Final model')
        )
    fig.legend(handles=legend_handles, ncol=3, frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(0.56, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        fig.savefig('r2_by_model.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_r2_over_time(cv_df, save=False):
    fold_time_r2 = (
        cv_df.groupby(['model', 'fold', 'time'])
        .apply(_r2, include_groups=False)
        .reset_index(name='r2')
    )
    time_r2 = (
        fold_time_r2.groupby(['model', 'time'])['r2']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'r2_mean', 'std': 'r2_std'})
    )

    all_models = sorted(cv_df['model'].unique())
    colors = dict(zip(all_models, MODEL_PALETTE[:len(all_models)]))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 1.2))

    for model in all_models:
        mdf = time_r2[time_r2['model'] == model].sort_values('time')
        color = colors[model]
        ax.plot(mdf['time'], mdf['r2_mean'], color=color, marker='o', markersize=5,
                linewidth=1.6, label=MODEL_PRETTY.get(model, model), zorder=3)
        ax.fill_between(mdf['time'],
                        mdf['r2_mean'] - mdf['r2_std'],
                        mdf['r2_mean'] + mdf['r2_std'],
                        alpha=0.10, color=color)

    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('R²', fontsize=10, labelpad=6)
    title = r"$\mathbf{R^2\ over\ time}$"
    ax.set_title(f"{title}\n(CV results)", fontsize=12, pad=12)
    ax.xaxis.grid(True, linestyle='-', alpha=0.5, linewidth=0.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.5, linewidth=0.5)
    _apply_spine_style(ax)

    ax.legend(frameon=False, fontsize=9, ncol=3,
        loc='upper center', bbox_to_anchor=(0.45, -0.15))
        
    plt.tight_layout()
    if save:
        fig.savefig('r2_over_time.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_feature_importance(imp_df, model, top_n=20, save=False):
    mdf = imp_df[imp_df['model'] == model]
    feat_stats = (
        mdf.groupby('feature')['importance_mean']
        .agg(['mean', 'std'])
        .sort_values('mean')
        .tail(top_n)
    )

    grp = 'Linear' if model in LINEAR_MODELS else 'Non-Linear'
    fc, ec = FILL_COLOR[grp], EDGE_COLOR[grp]
    fc_rgba = to_rgba(fc)
    fc_light = tuple(min(1, ch + 0.15) for ch in fc_rgba[:3]) + (0.85,)
    fc_dark  = tuple(max(0, ch - 0.08) for ch in fc_rgba[:3]) + (0.85,)

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(feat_stats) * 0.3)))

    for i, (feat, row) in enumerate(feat_stats.iterrows()):
        bar_fc = fc_light if row['mean'] >= 0 else fc_dark
        ax.barh(i, row['mean'], height=0.6, color=bar_fc, edgecolor=ec, linewidth=0.7, zorder=3)
        ax.errorbar(row['mean'], i, xerr=row['std'], fmt='none',
                    color=ec, linewidth=0.8, capsize=2, zorder=4)

    ax.set_yticks(range(len(feat_stats)))
    ax.set_yticklabels(feat_stats.index, fontsize=9)
    xlabel = 'Coefficient value' if grp == 'Linear' else 'Mean permutation importance'
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='#808080', linewidth=0.7)
    _apply_spine_style(ax)
    title = r"$\mathbf{Feature\ Importance}$"
    ax.set_title(f"{title}\n({MODEL_PRETTY.get(model, model)})", fontsize=10, pad=6)

    plt.tight_layout()
    if save:
        fig.savefig(f'importance_{model}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
