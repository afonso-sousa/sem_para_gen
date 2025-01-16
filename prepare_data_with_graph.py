import argparse
import os
import re

from datasets import load_dataset

from pseudo_semantic_graph import SemanticGraph


def remove_outer_quotes(sentence):
    if (sentence.startswith("'") and sentence.endswith("'")) or (
        sentence.startswith('"') and sentence.endswith('"')
    ):
        return sentence[1:-1]
    else:
        return sentence


def remove_outer_quotes_from_sample(sample):
    if "source" in sample:
        sample["source"] = remove_outer_quotes(sample["source"])
    if "target" in sample:
        sample["target"] = remove_outer_quotes(sample["target"])
    return sample


def remove_whitespaces(sample):
    source = sample.pop("source")
    target = sample.pop("target")
    # remove leading and trailing spaces
    source = source.strip()
    # remove double spaces
    source = re.sub(" +", " ", source)
    target = target.strip()
    target = re.sub(" +", " ", target)
    return {"source": source, "target": target, **sample}


def normalize_special_chars(sample):
    source = sample.pop("source")
    target = sample.pop("target")
    output = {"source": source, "target": target}
    for k, v in output.items():
        output[k] = (
            v.replace("``", '"')
            .replace("` `", '"')
            .replace("‘‘", '"')
            .replace("‘ ‘", '"')
            .replace("’’", '"')
            .replace("’ ’", '"')
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("''", '"')
            .replace("' '", '"')
        )
    return {**output, **sample}


def clean_data(sample):
    sample = remove_whitespaces(sample)
    sample = normalize_special_chars(sample)
    sample = remove_outer_quotes_from_sample(sample)
    return sample


def add_graphs_to_entries(sample):
    sample = clean_data(sample)
    try:
        graph = SemanticGraph.from_text(sample["source"])
    except:
        breakpoint()
    if (
        graph is None
        or len(graph.nodes) == 0
        or len(graph.nodes) < len(sample["source"].split()) / 3
    ):
        return {key: None for key in {**sample, "nodes": None, "edges": None}.keys()}
    node_dicts = [node.__dict__ for node in graph.nodes]
    edge_dicts = [
        {"start": start, "end": end, "relation": relation}
        for start, end, relation in graph.edges
    ]
    output_dict = {"nodes": node_dicts, "edges": edge_dicts}

    return {**sample, **output_dict}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphs.")
    # Add the command-line arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="raw-data/paws",
        help="Dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pairs_evals.csv",
        help="Output file.",
    )
    parser.add_argument(
        "--drop_exemplars",
        action="store_true",
        help="Whether to drop exemplars if exist",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(os.path.abspath(os.path.dirname(args.output_path)), exist_ok=True)

    print(f"{args.dataset_name}")
    print(f"Output path: {args.output_path}")

    datasets = load_dataset(
        args.dataset_name, data_files={args.split: f"{args.split}.csv.gz"}
    )
    dataset = datasets[args.split]

    # dataset = dataset.select(range(32408, len(dataset)))

    # from datasets import Dataset

    # single_entry = [
    #     {
    #         "source": ' Purple Clover " describes the site as being for people who are " ... still curious , still cool , and still crazy after all these years .',
    #         "target": ' Purple Clover " describes the site as being for people who are " ... still curious , still cool , and still crazy after all these years .',
    #     }
    # ]

    # # Create a dataset with the single entry
    # dataset = Dataset.from_list(single_entry)

    if "Unnamed: 0" in dataset.column_names:
        dataset.remove_columns(["Unnamed: 0"])

    if args.drop_exemplars and "exemplar" in dataset.column_names:
        dataset.remove_columns(["exemplar"])

    dataset = dataset.map(
        add_graphs_to_entries,
        num_proc=1,
    )

    dataset = dataset.filter(
        lambda entry: all(value is not None for value in entry.values())
    )

    dataset.to_json(args.output_path)


if __name__ == "__main__":
    main()
