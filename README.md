# Pseudo-Semantic Graphs for Generating Paraphrases


This repo contains the code for the paper [Pseudo-Semantic Graphs for Generating Paraphrases](https://link.springer.com/chapter/10.1007/978-3-031-73503-5_18), by Afonso Sousa & Henrique Lopes Cardoso (EPIA 2024).

Paraphrases are texts written using different words but conveying the same meaning; hence, their quality is based upon retaining
semantics while varying syntax/vocabulary. Recent works leverage structured syntactic information to control the syntax of the generations while relying on pretrained language models to retain the semantics. However, rarely do works in the literature consider using structured semantic information to enrich the language representation. In this work, we propose to model the task of paraphrase generation as a pseudo-Graph-to-Text task where we fine-tune pretrained language models using as input linearized representations of pseudo-semantic graphs built from dependency parsing trees sourced from the original input texts. Our model achieves competitive results on three popular paraphrase generation benchmarks.

## Installation
First, create a fresh conda environment and install the dependencies:
```
conda create -n [ENV_NAME] python=3.9
pip install -r requirements.txt
```

Additionally, you need to install the Spacy and coreference models:

```
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
python -m coreferee install en
```

## Citation

```
@inproceedings{10.1007/978-3-031-73503-5_18,
author = {Sousa, Afonso and Lopes Cardoso, Henrique},
title = {Pseudo-Semantic Graphs for Generating Paraphrases},
year = {2024},
isbn = {978-3-031-73502-8},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-73503-5_18},
doi = {10.1007/978-3-031-73503-5_18},
booktitle = {Progress in Artificial Intelligence: 23rd EPIA Conference on Artificial Intelligence, EPIA 2024, Viana Do Castelo, Portugal, September 3–6, 2024, Proceedings, Part III},
pages = {215–227},
numpages = {13},
keywords = {paraphrase generation, semantic graph, pretrained language model, dependency parsing},
location = {Viana do Castelo, Portugal}
}
```