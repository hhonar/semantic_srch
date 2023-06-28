# semantic_srch
This repository  is written to perform semantic search on the documents.
It contains code and resources on how to leverage PyTorch to perform efficient and accurate semantic search on a give corpus of documents.

## Overview:
Semantic search is a technique used to improve the relevance of search results by understanding the context and meaning of the query and the documents being searched.  This repository provides an implementation of semantic search using the all-MiniLM-L6-v2.  Without the loss of generality other models could be used.  Here the fine tuning of the model is not discussed.  However, this is possible and I have a separate repository for this purpose. 

## Requirements:
json/
pandas/
json_normalize/
re/
torch/
SentenceTransformer/
io/

## Usage
The input parameters has to be defined or given such as document, extracted text from OCR json file, number of top search results to be shown, the length of the sentencesdepending on how informative the outcome might be in the business usecase, keyphrases of interest and model name.
