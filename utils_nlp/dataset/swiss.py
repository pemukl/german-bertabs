# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

"""
    Utility functions for downloading, extracting, and reading the
    Swiss dataset at https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN.

"""

import nltk

# nltk.download("punkt")
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import regex as re
from torchtext.utils import extract_archive
import pandas
from sklearn.model_selection import train_test_split

from utils_nlp.dataset.url_utils import (
    maybe_download,
    maybe_download_googledrive,
    extract_zip,
)
from utils_nlp.models.transformers.datasets import (
    SummarizationDataset,
    IterableSummarizationDataset,
)


def _target_sentence_tokenization(line):
    return line.split("<q>")


def join(sentences):
    return " ".join(sentences)


def SwissSummarizationDataset(top_n=-1, validation=False):
    """Load the CNN/Daily Mail dataset preprocessed by harvardnlp group."""

    URLS = ["https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN/download?path=%2F&files=data_train.csv",
            "https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN/download?path=%2F&files=data_test.csv",]
    LOCAL_CACHE_PATH = '.data'

    FILE_NAME = "data_train.csv"
    maybe_download(URLS[0], FILE_NAME, LOCAL_CACHE_PATH)
    dataset_path = os.path.join(LOCAL_CACHE_PATH, FILE_NAME)
    
    train = pandas.read_csv(dataset_path).values.tolist()
    if(top_n!=-1):
        train = train[0:top_n]
    source = [item[0] for item in train]
    summary = [item[1] for item in train]
    train_source,test_source,train_summary,test_summary=train_test_split(source,summary,train_size=0.95,test_size=0.05,random_state=123)
    if validation:
        train_source, validation_source, train_summary, validation_summary = train_test_split(
            train_source, train_summary, train_size=0.9, test_size=0.1, random_state=123
        )
        return (
            SummarizationDataset(
                source_file=None,
                source=train_source,
                target=train_summary,
                source_preprocessing=[tokenize.sent_tokenize],
                target_preprocessing=[
                    tokenize.sent_tokenize,
                ],
                top_n=top_n,
            ),
            SummarizationDataset(
                source_file=None,
                source=validation_source,
                target=validation_summary,
                source_preprocessing=[tokenize.sent_tokenize],
                target_preprocessing=[
                    tokenize.sent_tokenize,
                ],
                top_n=top_n,
            ),
            SummarizationDataset(
                source_file=None,
                source=test_source,
                target=test_summary,
                source_preprocessing=[tokenize.sent_tokenize],
                target_preprocessing=[
                    tokenize.sent_tokenize,
                ],
                top_n=top_n,
            ),
        )
    else:
        return (
            SummarizationDataset(
                source_file=None,
                source=train_source,
                target=train_summary,
                source_preprocessing=[tokenize.sent_tokenize],
                target_preprocessing=[
                    tokenize.sent_tokenize,
                ],
                top_n=top_n,
            ),
            SummarizationDataset(
                source_file=None,
                source=test_source,
                target=test_summary,
                source_preprocessing=[tokenize.sent_tokenize],
                target_preprocessing=[
                    tokenize.sent_tokenize,
                ],
                top_n=top_n,
            ),
        )

