# %%
import pandas as pd
from datasets import load_dataset

dataset_name = "paranmt-small"
standard_path = f"output/facebook/bart-base-{dataset_name}-lr1e-4-standard/tagged_metrics.csv"
graph_path = (
    f"output/facebook/bart-base-{dataset_name}-lr1e-4-graph/tagged_metrics.csv"
)
graph_with_tokens_path = f"output/facebook/bart-base-{dataset_name}-lr1e-4-with_tokens/tagged_metrics.csv"

# qcpg_path = f"output/facebook/bart-base-{dataset_name}-lr1e-4-vector/tagged_metrics.csv"

# hrq_vae_path = f"output/hrq-vae/{dataset_name}/tagged_metrics.csv"

standard_dataset = load_dataset(
    "csv",
    data_files=standard_path,
    delimiter="\t",
)["train"]

graph_dataset = load_dataset(
    "csv",
    data_files=graph_path,
    delimiter="\t",
)["train"]

graph_with_tokens_dataset = load_dataset(
    "csv",
    data_files=graph_with_tokens_path,
    delimiter="\t",
)["train"]

# qcpg_dataset = load_dataset(
#     "csv",
#     data_files=qcpg_path,
#     delimiter="\t",
# )["train"]

# hrq_vae_dataset = load_dataset(
#     "csv",
#     data_files=hrq_vae_path,
#     delimiter="\t",
# )["train"]

# %%
metrics_datasets = [
    standard_dataset,
    graph_dataset,
    graph_with_tokens_dataset,
    # qcpg_dataset,
    # hrq_vae_dataset,
]
metric_names = ["ibleu", "sbert"]

sum_diffs = {metric: [] for metric in metric_names}

for metric in metric_names:
    num_datasets = len(metrics_datasets)
    sum_diff = [0] * len(metrics_datasets[0][metric])

    for i in range(num_datasets):
        for j in range(i + 1, num_datasets):
            metric_values_i = metrics_datasets[i][metric]
            metric_values_j = metrics_datasets[j][metric]
            diff = [
                abs(a - b) if a != 0 and b != 0 and (a - b) != 0 else 0
                for a, b in zip(metric_values_i, metric_values_j)
            ]
            sum_diff = [sum(x) for x in zip(sum_diff, diff)]

    sum_diffs[metric] = sum_diff

# Create a DataFrame with the computed sums of absolute differences
df = pd.DataFrame(sum_diffs)

# Sort by the largest sum of absolute differences
sorted_df = df.sort_values(by=["ibleu", "sbert"], ascending=False)

print(sorted_df)


# %%
selected_index = 49

print(
    f'Standard Sentence: {standard_dataset["prediction"][selected_index]} ({standard_dataset["ibleu"][selected_index]} - {standard_dataset["sbert"][selected_index]})'
)
print(
    f'Graph Sentence: {graph_dataset["prediction"][selected_index]} ({graph_dataset["ibleu"][selected_index]} - {graph_dataset["sbert"][selected_index]})'
)
print(
    f'Graph with Syntax Sentence: {graph_with_tokens_dataset["prediction"][selected_index]} ({graph_with_tokens_dataset["ibleu"][selected_index]} - {graph_with_tokens_dataset["sbert"][selected_index]})'
)
# print(
#     f'QCPG Sentence: {qcpg_dataset["prediction"][selected_index]} ({qcpg_dataset["ibleu"][selected_index]} - {qcpg_dataset["sbert"][selected_index]})'
# )
# print(
#     f'HRQ-VAE Sentence: {hrq_vae_dataset["prediction"][selected_index]} ({hrq_vae_dataset["ibleu"][selected_index]} - {hrq_vae_dataset["sbert"][selected_index]})'
# )
print("#" * 10)
print("Gold Standard: ", standard_dataset["target"][selected_index])
print("Source: ", standard_dataset["source"][selected_index])

# %%
