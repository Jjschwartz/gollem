# data

This directory contains scripts for downloading and processing different datasets we use for training. The idea is that for each dataset we have a `.py` script that (1) downloads the dataset, and (2) tokenizes the dataset and saves to a `.bin` file. Each script will create a new subdirectory that where all the downloaded and processed files are stored.

## Reference

Much of the code is based on the following references:

- [Karpathy's llm.c](https://github.com/karpathy/llm.c/blob/master/dev/data/README.md)