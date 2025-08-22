from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, re, math

# get descriptive stats (mean and SEM)
def get_stats_for_metrics(file_path: str):
    """
    Load JSON files (for 1 or more RAG outputs) and return:
      • detailed_df  – one row per query with row-wise averages
      • summary_df   – grand means + SEMs for plotting
    """
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detailed_df = pd.DataFrame(data)
    n_rows = len(detailed_df)

    # row-wise means for each RAG triad metric
    context_relevancy_cols = [c for c in detailed_df.columns if re.match(r"contextual_relevancy_no_", c)]
    answer_relevancy_cols = [c for c in detailed_df.columns if re.match(r"answer_relevancy_no_", c)]
    faithfulness_cols = [c for c in detailed_df.columns if re.match(r"faithfulness_no_",   c)]

    detailed_df["Contextual_Relevancy"] = detailed_df[context_relevancy_cols]  .mean(axis=1)
    detailed_df["Answer_Relevancy"] = detailed_df[answer_relevancy_cols]  .mean(axis=1)
    detailed_df["Faithfulness"] = detailed_df[faithfulness_cols].mean(axis=1)

    # means for latency and word count (as a measure for input token costs)
    detailed_df["Latency"] = detailed_df["latency"]/3
    detailed_df["Word_Count"] = detailed_df["word_count"]

    # store means + SEM
    summary = {
        "Contextual_Relevancy": detailed_df["Contextual_Relevancy"].mean(),
        "Answer_Relevancy": detailed_df["Answer_Relevancy"].mean(),
        "Faithfulness": detailed_df["Faithfulness"].mean(),
        "Latency": detailed_df["Latency"].mean(),
        "Word_Count": detailed_df["Word_Count"].mean(),
        # standard error of the mean
        "Contextual_Relevancy_SE": detailed_df["Contextual_Relevancy"].std(ddof=1) / math.sqrt(n_rows),
        "Answer_Relevancy_SE": detailed_df["Answer_Relevancy"].std(ddof=1) / math.sqrt(n_rows),
        "Faithfulness_SE": detailed_df["Faithfulness"].std(ddof=1) / math.sqrt(n_rows),
        "Latency_SE": detailed_df["Latency"].std(ddof=1) / math.sqrt(n_rows),
        "Word_Count_SE": detailed_df["Word_Count"].std(ddof=1) / math.sqrt(n_rows),
    }

    summary_df = pd.DataFrame([summary])
    return detailed_df, summary_df

# plot RAG metrics (RAG triad + latency + word count) with dots for individual queries
def plot_rag_metrics(metrics_dfs: dict[str, pd.DataFrame],
                               detailed_dfs: dict[str, pd.DataFrame],
                               figsize=(11, 7)):
    methods   = list(metrics_dfs.keys())
    n_methods = len(methods)
    colors    = plt.cm.Set2(np.linspace(0, 1, n_methods))
    bar_w     = 0.8 / n_methods          

    fig = plt.figure(figsize=figsize)
    gs  = plt.GridSpec(2, 2, height_ratios=[2, 1.3], hspace=0.35, wspace=0.30)

    # RAG Triade
    triad     = [("Contextual_Relevancy", "Context\nRelevance"),
                 ("Answer_Relevancy",     "Answer\nRelevance"),
                 ("Faithfulness",         "Faith-\nfulness")]
    x         = np.arange(len(triad))
    ax_top    = fig.add_subplot(gs[0, :])

    for m_idx, method in enumerate(methods):
        summ = metrics_dfs[method]
        det  = detailed_dfs[method]

        means  = [summ[m].iloc[0]          for m, _ in triad]
        errors = [summ[f"{m}_SE"].iloc[0]  for m, _ in triad]

        # Bars
        ax_top.bar(x + m_idx*bar_w, means, yerr=errors,
                   width=bar_w, color=colors[m_idx], capsize=4,
                   label=method)

        # Dots with jitter
        jitter = (np.random.rand(len(det)) - 0.5) * bar_w*0.2
        for j, (metric, _) in enumerate(triad):
            y_vals = det[metric]
            ax_top.scatter(np.full_like(y_vals, x[j] + m_idx*bar_w) + jitter,
                        y_vals,
                        s=18,
                        alpha=0.2,
                        edgecolors=(0, 0, 0, 0.2),   
                        facecolors=colors[m_idx])

    ax_top.set_ylim(0, 1.2)
    ax_top.set_ylabel("Score")
    ax_top.set_xticks(x + bar_w*(n_methods-1)/2)
    ax_top.set_xticklabels([lab for _, lab in triad])
    ax_top.grid(axis="y", linestyle="--", alpha=0.6)

    yticks = ax_top.get_yticks()
    ytick_labels = [f'{y:.1f}' if y <= 1.0 else '' for y in yticks]
    ax_top.set_yticklabels(ytick_labels)

    ax_top.legend()

    # Latency
    ax_lat = fig.add_subplot(gs[1, 0])

    for m_idx, method in enumerate(methods):
        summ = metrics_dfs[method]
        det  = detailed_dfs[method]

        mean = summ["Latency"].iloc[0]
        err  = summ["Latency_SE"].iloc[0]

        ax_lat.bar(m_idx, mean, yerr=err,
                   width=bar_w * n_methods,      
                   color=colors[m_idx], capsize=4)

        ax_lat.scatter(np.full_like(det["Latency"], m_idx) + jitter,
                       det["Latency"],
                       s=18,
                       alpha=0.2,
                       edgecolors=(0, 0, 0, 0.2),
                       facecolors=colors[m_idx])

    ax_lat.set_xticks(range(n_methods))
    ax_lat.set_xticklabels(methods)
    ax_lat.set_ylabel("Seconds")
    ax_lat.set_title("Latency")
    ax_lat.grid(axis="y", linestyle="--", alpha=0.6)

    # Word count
    ax_wc = fig.add_subplot(gs[1, 1])

    for m_idx, method in enumerate(methods):
        summ = metrics_dfs[method]
        det  = detailed_dfs[method]

        mean = summ["Word_Count"].iloc[0]
        err  = summ["Word_Count_SE"].iloc[0]

        ax_wc.bar(m_idx, mean, yerr=err,
                  width=bar_w * n_methods,
                  color=colors[m_idx], capsize=4)

        ax_wc.scatter(np.full_like(det["Word_Count"], m_idx) + jitter,
                      det["Word_Count"],
                      s=18,
                       alpha=0.2,
                       edgecolors=(0, 0, 0, 0.2),
                      facecolors=colors[m_idx])

    ax_wc.set_xticks(range(n_methods))
    ax_wc.set_xticklabels(methods)
    ax_wc.set_ylim(0, 800)
    ax_wc.set_ylabel("Words")
    ax_wc.set_title("Word Count")
    ax_wc.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
