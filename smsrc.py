# -------------------------------------------------------------
# Script written for ESG Bond Framework
# Written by H. Honari  
# -------------------------------------------------------------


# importing packages
import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import re
import torch
from sentence_transformers import SentenceTransformer, util
import io




# Define the keys to generate the hierarchical dict
__docname__ = 'SAMPLEDOCUMENT'
pdf_json = 'structuredData.json'
n = 5 # number of instances [i.e. for top 5 search results]
sent_thrshold = 10 # lenght of sentence - very short not informative - check with client
# inputs for semantic search function and call
xcl_query_name = 'keyphrase1.xlsx'
mod_name = 'all-MiniLM-L6-v2'






# --------------------------------------------------------------
# functions
# --------------------------------------------------------------



# Define the function for directory
def set_dir():
    dpath = os.path.abspath(__file__)
    __dir__ = os.path.dirname(dpath)
    return __dir__



# Define a functino for creating a hierarchical dict
def hidict(n_inst, key_inst, keys0, keys1, keys2,main_key):
    lev2 = {k: [] for k in keys2}
    lev1 = {k: [] for k in keys1}
    for _ in range(n_inst):
        lev1[key_inst].append(lev2.copy())
    lev0 = {k: lev1 for k in keys0}
    Search = {main_key: lev0}
    return Search



# Semantic search:
def semsrch(__dir__,xcl_query_name,mod_name,filt_sub_corpus,filt_lib,Search,__docname__,keys0):
    # list of the queries provided - client verified
    dqueries = pd.read_excel (os.path.join(__dir__,'Files',xcl_query_name))
    __moddir__ = os.path.join(__dir__,mod_name)
    embedder = SentenceTransformer(__moddir__)
    corpus_embeddings = embedder.encode(filt_sub_corpus, convert_to_tensor=True)

    # Query sentences:
    queries = dqueries['Keyphrase'].iloc[0:]  #list of queries

    # Defining the keys 
    results = []
    keys = ['Page','Text','Score','filePaths']
    results.append(keys)

    # Find closest n sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(n, len(filt_sub_corpus))

    if type(queries) == pd.core.series.Series:
        for i, query in enumerate(queries):
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            
            # We use cosine-similarity and torch.topk to find the highest n scores
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            j = 0
            for score, idx in zip(top_results[0], top_results[1]):
                results.append([int(filt_lib['Page'].iloc[idx.item()]+1),filt_sub_corpus[idx],"(Score: {:.4f})".format(score), filt_lib['filePaths'].iloc[idx.item()]] )
                Search[__docname__][keys0[i]]['search'][j].update({'page': int(filt_lib['Page'].iloc[idx.item()]+1),
                                                                    'text': filt_sub_corpus[idx],
                                                                    'score': "{:.2f}".format(score)})
                j = j + 1
            if top_results[0][0].item() >= 0.2:
                    Search[__docname__][keys0[i]]['decision'] = 'Yes' 
            else:
                    Search[__docname__][keys0[i]]['decision'] = 'No'
            Search[__docname__][keys0[i]]['score'] = "{:.2f}".format(top_results[0][0].item())

    elif type(queries) == str:
        query = queries.strip('][').split('  ,  ')
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
    resultsDataFrame = pd.DataFrame(results, columns = keys)
    resultsDataFrame.to_excel(os.path.join(__dir__,'out',__docname__+'.xlsx'))
    with io.open(os.path.join(__dir__,'out',__docname__+'.json'), "w", encoding="utf-8") as outfile:
        json.dump(Search, outfile,ensure_ascii=False)
    return resultsDataFrame


# input function
def init_func(__docname__, pdf_json, mod_name,xcl_query_name, sent_thrshold,n):

    # Set file sub directory
    __dir__ = set_dir()
    __subdir__ = os.path.join(__dir__,'Files',__docname__,pdf_json)

    # Read json file from extracted pdf 
    with open(__subdir__,encoding="utf8") as json_data:
        data = json.load(json_data)

    # Create a dataFrame
    df1 = pd.DataFrame(data['elements'])

    # Create a corpus with necessary data fields, drop na
    lib_corpus= df1[df1['Text'].notna()]
    lib_corpus = lib_corpus[['Text','Page','filePaths']]

    # Create a list from the doc
    subject_corpus = lib_corpus['Text'].tolist()


    # Filter out short sentences, headers, sentences from table of contents
    # Note: Set a threshold for len of sentence - default 10 [business need]
    threshold = sent_thrshold # this is the threshold for the length of the sentences
    filt_sub_corpus = []
    lib_index = []

    for i in range(len(subject_corpus)):
        if (len(subject_corpus[i].split()) > threshold or re.findall(r'(https?://\S+)', subject_corpus[i])):
            filt_sub_corpus.append(subject_corpus[i])
            lib_index.append(i)
    filt_lib = lib_corpus.iloc[lib_index,:]

    # default keys for the search framework
    key_inst = 'search'
    keys0 = ['keyword1','keyword2','keyword3','keyword4','keyword5','keyword6','keyword7'] # keys
    keys2 = ['page', 'text', 'score']
    keys1 = ['search','score','decision']

    # call the search to create dict
    Search = hidict(n, key_inst, keys0, keys1, keys2, __docname__)

    # call semsearch to find results
    semsrch(__dir__,xcl_query_name,mod_name,filt_sub_corpus,filt_lib,Search,__docname__,keys0)

    return



# function for all docs directory
def src_all(__folders__):
     for __docname__ in __folders__:
          init_func(__docname__, pdf_json, mod_name,xcl_query_name, sent_thrshold,n)
          print(__docname__ + ' .... processed successfully!')
__dir__ = set_dir()   
__folders__ = [n for n in os.listdir(os.path.join(__dir__,'Files')) if os.path.isdir(os.path.join(os.path.join(__dir__,'Files'), n))]

# call the function
src_all(__folders__)


# call the function if want individual doc
#init_func(__docname__, pdf_json, mod_name,xcl_query_name, sent_thrshold,n)


# function to merge all jsons
def mrg_json(jdir,jout):
     mrged = {}
     __dir__ = set_dir()  
     for fpath in jdir:
          with open(os.path.join(__dir__,'out',fpath), 'r', encoding="utf-8") as j:
               df = json.load(j)
               mrged.update(df)
     with open(jout, "w", encoding="utf-8") as j:
        json.dump(mrged, j,ensure_ascii=False)

__dir__ = set_dir()   
__outdir__ = os.path.join(__dir__,'out')
json_files = [n for n in os.listdir(__outdir__) if n.endswith('.json')]

# call json merge:
mrg_json(json_files, os.path.join(__dir__,'out','merged_json_files'))
          