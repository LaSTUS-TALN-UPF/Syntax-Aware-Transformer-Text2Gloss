# Syntax-Aware-Transformer-Text2Gloss

This repository contains the code of the experiments reported in the paper "Syntax-aware Transformers for Neural Machine Translation: The Case ofText to Sign Gloss Translation".
The paper was accepted for the Workshop BUCC2021 in the conference RANLP 2021. 

## Abstract

It is well-established that the preferred mode of communication of the deaf and hard of hearing (DHH) community are Sign Languages (SLs), but they are considered low resource languages where natural language processing technologies are of concern. In this paper we study the problem of text to SL gloss Machine Translation (MT) using Transformer-based architectures. Despite the significant advances of MT for spoken languages in the recent couple of decades, MT is in its infancy when it comes to SLs.  We enrich a Transformer-based architecture aggregating syntactic information extracted from a dependency parser to word-embeddings. We test our model on a well-known dataset showing that the syntax-aware model obtains performance gains in terms of MT evaluation metrics. 

## Requirements
This research was developed using Python 3.8.0. Below, the library requirements are listed to assure the experiments reproducibility.

| Resource | Version/URL |
| ------------- | ------------- |
| Tensorflow | 2.4.1 |
| Numpy | 1.19.5 |
| Spacy | 3.0.5 |
| NLTK | 3.5 |
| SentencePiece | [LINK](https://github.com/google/sentencepiece) |
| PyTer | [LINK](https://github.com/BramVanroy/pyter) |
| ROUGE | [LINK](https://github.com/google/seq2seq/blob/master/seq2seq/metrics/rouge.py) |
| SacreBLEU | [LINK](https://github.com/mjpost/sacrebleu) |
 
## Citation


@inproceedings{
    egea-gomez-etal-2021-syntax,
    title = "Syntax-aware Transformers for Neural Machine Translation: The Case of Text to Sign Gloss Translation",
    author = "Egea G{\'o}mez, Santiago  and
      McGill, Euan  and
      Saggion, Horacio",
    booktitle = "Proceedings of the 14th Workshop on Building and Using Comparable Corpora (BUCC 2021)",
    month = sep,
    year = "2021",
    address = "Online (Virtual Mode)",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.bucc-1.4",
    pages = "18--27",
    abstract = "It is well-established that the preferred mode of communication of the deaf and hard of hearing (DHH) community are Sign Languages (SLs), but they are considered low resource languages where natural language processing technologies are of concern. In this paper we study the problem of text to SL gloss Machine Translation (MT) using Transformer-based architectures. Despite the significant advances of MT for spoken languages in the recent couple of decades, MT is in its infancy when it comes to SLs. We enrich a Transformer-based architecture aggregating syntactic information extracted from a dependency parser to word-embeddings. We test our model on a well-known dataset showing that the syntax-aware model obtains performance gains in terms of MT evaluation metrics.",
}

