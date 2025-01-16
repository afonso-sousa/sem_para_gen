from collections import defaultdict

from amr_utils import simplify_amr_nopar
from pseudo_semantic_graph import SemanticGraph


def linearize_amr(node_tokens, edge_triples):
    def topological_sort(node, visited, stack):
        visited[node] = True
        if node in graph:
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    topological_sort(neighbor, visited, stack)
        stack.append(node)

    graph = defaultdict(list)
    for edge in edge_triples:
        if edge[2] == 0:  # Only consider edges with the last index equal to 0
            graph[edge[0]].append(edge[1])

    visited = {i: False for i in range(len(node_tokens))}
    stack = []

    for node in range(len(node_tokens)):
        if not visited[node]:
            topological_sort(node, visited, stack)

    linearized_nodes = [node_tokens[i] for i in reversed(stack)]

    return linearized_nodes


def chain_dep_trees(dep_trees):
    total_nodes = 0
    chained_dep_tree = []

    for dep_tree in dep_trees:
        adjusted_dep_tree = []

        for node in dep_tree:
            adjusted_node = node.copy()
            if adjusted_node["head"] != -1:
                adjusted_node["head"] += total_nodes
            adjusted_dep_tree.append(adjusted_node)

        total_nodes += len(dep_tree)
        chained_dep_tree.extend(adjusted_dep_tree)

    return chained_dep_tree


def linearize_dep_trees(all_nodes):
    def _find_roots(tree):
        roots = []
        for idx, token in enumerate(tree):
            if token["dep"] == "ROOT":
                roots.append(idx)
        return roots

    def _get_children(tree, node_idx):
        children = []
        for idx, token in enumerate(tree):
            if token["head"] == node_idx:
                children.append(idx)
        return children

    def _traverse(tree, current_node_idx, visited):
        result = []
        # Check if the node has been visited
        if current_node_idx in visited:
            return result

        # Add the current node to the visited list
        visited.append(current_node_idx)

        children = _get_children(tree, current_node_idx)
        relation = tree[current_node_idx]["dep"].upper()
        relation_str = f":{relation} " if tree[current_node_idx]["head"] != -1 else ""

        children_output = []
        for child in children:
            children_output += [_traverse(tree, child, visited)]

        children_output = " ".join(children_output)
        children_output = " " + children_output if children_output else ""

        result = f"{relation_str}{tree[current_node_idx]['token']}" + children_output

        return result

    result = ""
    for root_idx in _find_roots(all_nodes):
        output = _traverse(all_nodes, root_idx, [])
        result = result + " " + output

    return result


def processing_function_for_semantic_graph(
    tokenizer,
    max_source_length,
    max_target_length,
    with_token_types,
    return_sources=False,
    reentrancy_tokens=False,
    delimiters=False,
    relations=False,
):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        # requires examples to be single entry
        assert not isinstance(
            inputs, list
        ), f"You have a batch of {len(inputs)} examples. \
            The linearization of the graph is done instance by instance. \
            Please do not use `batched` examples."

        try:
            g = SemanticGraph(nodes=examples["nodes"], edges=examples["edges"])
            linearized_token_list = g.linearize(
                reentrancy_tokens=reentrancy_tokens,
                delimiters=delimiters,
                relations=relations,
            )

            linearized_types, linearized_tokens = list(zip(*linearized_token_list))
        except:
            breakpoint()

        # TODO add special tokens to BART tokenizer
        # graph_inputs = (
        #     [tokenizer.bos_token] + list(linearized_tokens) + [tokenizer.eos_token]
        # )
        graph_inputs = list(linearized_tokens) + [tokenizer.eos_token]

        model_inputs = tokenizer(
            graph_inputs,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            is_split_into_words=True,
        )

        if with_token_types:
            types = [1] + list(linearized_types) + [1]

            token_label_ids = []
            for word_idx in model_inputs.word_ids():
                if word_idx is None:
                    breakpoint()
                token_label_ids.append(types[word_idx])

            model_inputs["token_label_ids"] = token_label_ids

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        # if return_sources:
        #     raw_inputs = tokenizer(
        #         inputs,
        #         max_length=max_source_length,
        #         padding=False,
        #         truncation=True,
        #     )
        #     model_inputs["raw_inputs"] = raw_inputs["input_ids"]

        return model_inputs

    return preprocess_function


def standard_processing_function(
    tokenizer,
    max_source_length,
    max_target_length,
):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function


def processing_function_for_AMR(
    tokenizer,
    max_source_length,
    max_target_length,
):
    def preprocess_function(examples):
        # transliterate Unicode text into its closest ASCII representation
        inputs = examples["source"]
        targets = examples["target"]

        # requires examples to be single entry
        assert not isinstance(
            inputs, list
        ), f"You have a batch of {len(inputs)} examples. \
            The linearization of the graph is done instance by instance. \
            Please do not use `batched` examples."

        if not examples["node_tokens"]:
            return {"input_ids": None, "attention_mask": None, "labels": None}
        inputs = linearize_amr(examples["node_tokens"], examples["edge_triples"])
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function


def processing_function_for_dep_trees(
    tokenizer,
    max_source_length,
    max_target_length,
):
    def preprocess_function(examples):
        # transliterate Unicode text into its closest ASCII representation
        inputs = examples["source"]
        targets = examples["target"]

        # requires examples to be single entry
        assert not isinstance(
            inputs, list
        ), f"You have a batch of {len(inputs)} examples. \
            The linearization of the graph is done instance by instance. \
            Please do not use `batched` examples."

        inputs_list = linearize_dep_trees(
            chain_dep_trees(examples["src_dep_trees"])
        ).split()

        model_inputs = tokenizer(
            inputs_list,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function


def parse_amr_into_reduced_form(example):
    amr = simplify_amr_nopar(example["src_amr"])

    if amr is None:
        return {**example, "node_tokens": None, "edge_triples": None}

    nodes, triples = amr
    converted_triples = [
        (from_idx, to_idx, 0 if relation == "d" else 1)
        for from_idx, to_idx, relation in triples
    ]
    return {**example, "node_tokens": nodes, "edge_triples": converted_triples}
