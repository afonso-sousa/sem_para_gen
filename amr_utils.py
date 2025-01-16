"""
This file has utils functions to extract an AMR graph from a sentence.
It is based on the code from https://github.com/zzshou/AMRSim/blob/main/preprocess/utils.py
"""

import amrlib
import penman


def simplify_nopar(tokens, v2c):
    mapping = {}
    new_tokens = []
    for tok in tokens:
        # ignore instance-of
        if tok.startswith("("):
            # new_tokens.append('(')
            last_map = tok.replace("(", "")
            continue
        elif tok == "/":
            save_map = True
            continue
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(":"):
            new_tok = tok.strip(")")
            new_tok = new_tok.split("~")[0]
            new_tokens.append(new_tok)

        # concepts/reentrancies, treated similar as above
        else:
            new_tok = tok.strip(")")
            new_tok = new_tok.split("~")[0]

            if new_tok == "":
                continue

            # now we check if it is a concept or a variable (reentrancy)
            if new_tok in v2c:
                # reentrancy: replace with concept
                if new_tok not in mapping:
                    mapping[new_tok] = set()
                mapping[new_tok].add(len(new_tokens))

                if v2c[new_tok] is not None:
                    new_tok = v2c[new_tok]

            # check number
            elif new_tok.isnumeric():
                new_tok = new_tok

            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':
                new_tok = new_tok[1:-1]

            if new_tok != "":
                new_tokens.append(new_tok)

            if save_map:
                if last_map not in mapping:
                    mapping[last_map] = set()

                mapping[last_map].add(len(new_tokens) - 1)
                save_map = False

    return new_tokens, mapping


def get_positions(new_tokens, src):
    pos = []
    for idx, n in enumerate(new_tokens):
        if n == src:
            pos.append(idx)
    return pos


def get_line_amr_graph(graph, new_tokens, mapping, roles_in_order):
    triples = []
    nodes_to_print = new_tokens

    graph_triples = graph.triples

    edge_id = -1
    triples_set = set()
    count_roles = 0
    for triple in graph_triples:
        src, edge, tgt = triple
        if edge == ":instance" or edge == ":instance-of":
            continue

        # if penman.layout.appears_inverted(graph_penman, v):
        if (
            "-of" in roles_in_order[count_roles]
            and "-off" not in roles_in_order[count_roles]
        ):
            if edge != ":consist-of":
                edge = edge + "-of"
                old_tgt = tgt
                tgt = src
                src = old_tgt

        assert roles_in_order[count_roles] == edge

        count_roles += 1

        if edge == ":wiki":
            continue

        src = str(src).replace('"', "")
        tgt = str(tgt).replace('"', "")

        if src not in mapping:
            src_id = get_positions(new_tokens, src)
        else:
            src_id = sorted(list(mapping[src]))
        # check edge to verify
        edge_id = get_edge(new_tokens, edge, edge_id)

        if tgt not in mapping:
            tgt_id = get_positions(new_tokens, tgt)
        else:
            tgt_id = sorted(list(mapping[tgt]))

        for s_id in src_id:
            if (s_id, edge_id, "d") not in triples_set:
                triples.append((s_id, edge_id, "d"))
                triples_set.add((s_id, edge_id, "d"))
                triples.append((edge_id, s_id, "r"))
        for t_id in tgt_id:
            if (edge_id, t_id, "d") not in triples_set:
                triples.append((edge_id, t_id, "d"))
                triples_set.add((edge_id, t_id, "d"))
                triples.append((t_id, edge_id, "r"))

    if nodes_to_print == []:
        # single node graph, first triple is ":top", second triple is the node
        triples.append((0, 0, "s"))
    return nodes_to_print, triples


def get_edge(tokens, edge, edge_id):
    for idx in range(edge_id + 1, len(tokens)):
        if tokens[idx] == edge:
            return idx


def create_set_instances(graph_penman):
    instances = graph_penman.instances()
    dict_insts = {}
    for i in instances:
        dict_insts[i.source] = i.target
    return dict_insts


def simplify_amr_nopar(amr_str):
    try:
        graph_penman = penman.decode(amr_str)
        v2c_penman = create_set_instances(graph_penman)

        linearized_amr = penman.encode(graph_penman)
        linearized_amr = linearized_amr.replace("\t", "")
        linearized_amr = linearized_amr.replace("\n", "")
        tokens = linearized_amr.split()

        new_tokens, mapping = simplify_nopar(tokens, v2c_penman)

        roles_in_order = []
        for token in tokens:
            if token.startswith(":"):
                if token == ":instance-of":
                    continue
                roles_in_order.append(token)

        nodes, triples = get_line_amr_graph(
            graph_penman, new_tokens, mapping, roles_in_order
        )
        triples = sorted(triples)

        return nodes, triples
    except:
        return None


def get_amr_graph_for_sentences(sentences):
    # Load pretrained StoG model and generate AMR
    stog = amrlib.load_stog_model(
        model_dir="model_parse_xfm_bart_large-v0_1_0", device="cuda:0", batch_size=4
    )
    if isinstance(sentences, str):
        sentences = [sentences]
    gen_amr_strs = stog.parse_sents(sentences)
    nodes_batch, triples_batch = [], []
    for amr_str in gen_amr_strs:
        gen_amr_str = amr_str.split("\n", 1)[1]

        # Convert String AMR to list of nodes and triples
        amr = simplify_amr_nopar(gen_amr_str)
        if amr is None:
            return None

        nodes, triples = amr

        nodes_batch.append(nodes)
        triples_batch.append(triples)

    return nodes_batch, triples_batch


def get_amr_for_sentences(sentences):
    # Load pretrained StoG model and generate AMR
    stog = amrlib.load_stog_model(
        model_dir="model_parse_xfm_bart_large-v0_1_0", device="cuda:0", batch_size=4
    )
    if isinstance(sentences, str):
        sentences = [sentences]
    gen_amr_strs = stog.parse_sents(sentences)
    amr_strings = []
    for amr_str in gen_amr_strs:
        gen_amr_str = amr_str.split("\n", 1)[1]
        amr_strings.append(gen_amr_str)

    return amr_strings


if __name__ == "__main__":
    # sentences = [
    #     "The dog is walking.",
    #     "The dog is walking to the park.",
    #     "The dog is walking to the park because it is sunny.",
    # ]
    # # nodes, triples = get_amr_graph_for_sentences(sentences)
    # # print(nodes)
    # # print(triples)
    # amr_strings = get_amr_for_sentences(sentences)
    # print(amr_strings)

    amr_str = '(ii / inspect-01\n      :ARG0 (a / and\n            :op1 (p / person\n                  :name (n / name\n                        :op1 "Hagop"\n                        :op2 "Khajirian")\n                  :ARG0-of (h / have-org-role-91\n                        :ARG1 (o / organization\n                              :name (n2 / name\n                                    :op1 "FIBA"\n                                    :op2 "Asia"))\n                        :ARG2 (s / secretary\n                              :mod (d / deputy)\n                              :mod (g / general))))\n            :op2 (p2 / person\n                  :name (n3 / name\n                        :op1 "Manuel"\n                        :op2 "V."\n                        :op3 "Pangilinan")\n                  :ARG0-of (h2 / have-org-role-91\n                        :ARG1 (o2 / organization\n                              :name (n4 / name\n                                    :op1 "SBP"))\n                        :ARG2 (p3 / president))))\n      :ARG1 (v / venue)\n      :time (d2 / date-entity\n            :month 1\n            :year 2011))'

    print(amr_str)
    print(simplify_amr_nopar(amr_str))

# import json

# # %%
# import penman

# # amr_str = '(ii \/ inspect-01\n      :ARG0 (a \/ and\n            :op1 (p \/ person\n                  :name (n \/ name\n                        :op1 "Hagop"\n                        :op2 "Khajirian")\n                  :ARG0-of (h \/ have-org-role-91\n                        :ARG1 (o \/ organization\n                              :name (n2 \/ name\n                                    :op1 "FIBA"\n                                    :op2 "Asia"))\n                        :ARG2 (s \/ secretary\n                              :mod (d \/ deputy)\n                              :mod (g \/ general))))\n            :op2 (p2 \/ person\n                  :name (n3 \/ name\n                        :op1 "Manuel"\n                        :op2 "V."\n                        :op3 "Pangilinan")\n                  :ARG0-of (h2 \/ have-org-role-91\n                        :ARG1 (o2 \/ organization\n                              :name (n4 \/ name\n                                    :op1 "SBP"))\n                        :ARG2 (p3 \/ president))))\n      :ARG1 (v \/ venue)\n      :time (d2 \/ date-entity\n            :month 1\n            :year 2011))'
# # amr_str = "(w / walk-01\n      :ARG0 (d / dog))"


# file_path = "processed-data/paws/test_with_amr.jsonl"

# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         data = json.loads(line)
#         break

# amr_str = data["src_amr"]
# # %%

# # %%
# simplify_amr_nopar(amr_str)
# # %%
