import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from bart_types import TokenBartForConditionalGeneration
from pseudo_semantic_graph import SemanticGraph


def load_model(model_path, with_token_types=False):
    config = AutoConfig.from_pretrained(
        model_path,
    )
    if with_token_types:
        model = TokenBartForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=config,
        )
    model.eval()
    return model


# model_path = "output/facebook/bart-base-qqppos-lr1e-4-standard"
model_path = "output/facebook/bart-base-qqppos-lr1e-4-graph"
# model_path = "output/facebook/bart-base-qqppos-lr1e-4-with_tokens"
with_graph = True
with_token_types = False
reentrancy_tokens = True

max_source_length = 128

model = load_model(model_path, with_token_types)
tokenizer = AutoTokenizer.from_pretrained(model_path)

layer_num = 5  # Choose the layer you want to visualize
head_num = 0  # Choose the attention head you want to visualize

sentence = "Pat loves Chris."

if with_graph:
    graph = SemanticGraph.from_text(sentence)
    node_dicts = [node.__dict__ for node in graph.nodes]
    edge_dicts = [
        {"start": start, "end": end, "relation": relation}
        for start, end, relation in graph.edges
    ]

    g = SemanticGraph(nodes=node_dicts, edges=edge_dicts)
    linearized_token_list = g.linearize(reentrancy_tokens=reentrancy_tokens)

    linearized_types, linearized_tokens = list(zip(*linearized_token_list))

    graph_inputs = (
        [tokenizer.bos_token] + list(linearized_tokens) + [tokenizer.eos_token]
    )

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

else:
    model_inputs = tokenizer(
        sentence,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        add_special_tokens=True,
    )

input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0)
attention_mask = torch.tensor(model_inputs["attention_mask"]).unsqueeze(0)
if with_token_types:
    token_label_ids = torch.tensor(model_inputs["token_label_ids"]).unsqueeze(0)


# Forward pass
with torch.no_grad():
    if with_token_types:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_label_ids=token_label_ids,
            output_attentions=True,
        )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

# Extract attention scores
attention = outputs["encoder_attentions"]

print(f"Max layer num: {len(attention)}")
print(f"Max head num: {attention[0][0].size(0)}")

for l in range(len(attention)):
    for h in range(attention[0][0].size(0)):
        # Get the attention scores for the selected layer and head
        attention_scores = attention[l][0][h].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"])

        # Plot the attention scores
        fig, ax = plt.subplots(figsize=(len(tokens) * 0.5, len(tokens) * 0.5))
        plt.imshow(attention_scores, cmap="viridis")

        # Set y and x axis labels with the corresponding tokens
        ax.set_yticks(range(len(tokens)))
        ax.set_xticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=12)
        ax.set_xticklabels(tokens, fontsize=12, rotation=90)

        plt.title(
            f"Self-Attention Scores - Layer {l}, Head {h}",
            fontsize=14,
        )
        plt.colorbar()

        plt.savefig(
            f"attention_scores_l{l}_h{h}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
