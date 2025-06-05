#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gensim
import re
import nltk
import spacy
import string
from contractions import fix 
from nltk.corpus import stopwords
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


# ## TRACK 1: Discrete Text Representation

# Track 1: Choose a discrete representation method we have seen in class, such as n-gram word or character-level representations, Count Vectorizer, or TF-IDF. 
# 
# I decided to implement TF-IDF because it's the one I obtained better results with.

# In[2]:


train = pd.read_csv("train_responses.csv")
dev = pd.read_csv("dev_responses.csv")  
test = pd.read_csv("test_prompts.csv")
train_dev = pd.concat([train, dev], ignore_index=True)  


# In[3]:


train["user_prompt_clean"] = train["user_prompt"].astype(str).str.strip().str.lower()
dev["user_prompt_clean"] = dev["user_prompt"].astype(str).str.strip().str.lower()

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 5)) #initialize the vectorizer
train_tfidf = vectorizer.fit_transform(train["user_prompt_clean"]) #numerical representation of all prompts
dev_tfidf = vectorizer.transform(dev["user_prompt_clean"])

#cosine similarity between prompts in train and prompts in dev
similarity_matrix = cosine_similarity(dev_tfidf, train_tfidf)
best_match_indices = similarity_matrix.argmax(axis=1)
retrieved_responses = train["model_response"].iloc[best_match_indices].values
dev["retrieved_response"] = retrieved_responses #saving the retrieved responses in the dev dataset

smoothing_function = SmoothingFunction()

def bleu(x, y):
    if type(x) == str and type(y) == str:  
        return sentence_bleu([x.split()], y.split(), weights=(0.5, 0.5, 0, 0),  #only unigrams and bigrams
            smoothing_function=smoothing_function.method3)
    else:
        return 0.0  #if not string

dev["bleu_score"] = dev.apply(lambda x: bleu(x["model_response"], x["retrieved_response"]), axis=1)

print(f"Mean BLEU Score: {dev['bleu_score'].mean()}")
display(dev[["model_response", "retrieved_response", "bleu_score"]].head())


# In[4]:


test["user_prompt_clean"] = test["user_prompt"].astype(str).str.strip().str.lower()
train_dev["user_prompt_clean"] = train_dev["user_prompt"].astype(str).str.strip().str.lower()

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 5)) #initialize the vectorizer
train_dev_tfidf = vectorizer.fit_transform(train_dev["user_prompt_clean"])
test_tfidf = vectorizer.transform(test["user_prompt_clean"]) #numerical representation of all prompts

#cosine similarity between prompts in test and prompts in train_dev
similarity_matrix = cosine_similarity(test_tfidf, train_dev_tfidf)

best_match_indices = similarity_matrix.argmax(axis=1)
test["response_id"] = train_dev.iloc[best_match_indices]["conversation_id"].values

submission = test[["conversation_id", "response_id"]]
submission.to_csv("track_1_test.csv", index=False)
print("Done")


# ## TRACK 2: Distributed Static Text Representation

# Track 2: Choose a static distributed representation method we have seen in class, such as Word2vec, Doc2Vec, or pretrained embeddings like FastText.
# 
# I chose FastText as it is the one which had the best performance for me.

# In[5]:


train = pd.read_csv("train_responses.csv")
dev = pd.read_csv("dev_responses.csv")  
test = pd.read_csv("test_prompts.csv")
train_dev = pd.concat([train, dev], ignore_index=True)  


# In[6]:


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

#preprocessing of the data
def clean_text(text):
    text = text.lower()
    text = fix(text) #for instance can't becomes cannot
    text = re.sub(f"[{string.punctuation}]", "", text) #no punctuation
    text = re.sub(r"\d+", "", text) #no numbers
    text = re.sub(r"\s+", " ", text).strip() #no extra spaces
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct] #lemmatization
    return " ".join(words)

train["user_prompt_clean"] = train["user_prompt"].apply(clean_text)
dev["user_prompt_clean"] = dev["user_prompt"].apply(clean_text)


# In[7]:


fasttext_model = api.load("fasttext-wiki-news-subwords-300") #loading FastText

train["tokenized"] = train["user_prompt_clean"].apply(lambda x: x.split()) #tokenization
dev["tokenized"] = dev["user_prompt_clean"].apply(lambda x: x.split()) #tokenization

def sentence_embedding(tokens, model):
    word_vectors = [model[word] for word in tokens if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

train["embedding"] = train["tokenized"].apply(lambda tokens: sentence_embedding(tokens, fasttext_model))
dev["embedding"] = dev["tokenized"].apply(lambda tokens: sentence_embedding(tokens, fasttext_model))

train_embeddings = np.vstack(train["embedding"].values) #matrix form
dev_embeddings = np.vstack(dev["embedding"].values)

#compute cosine similarity
similarity_matrix = cosine_similarity(dev_embeddings, train_embeddings)
best_match_indices = similarity_matrix.argmax(axis=1)
retrieved_responses = train["model_response"].iloc[best_match_indices].values
dev["retrieved_response"] = retrieved_responses #retrieving all the responses and putting them in a new column in dev

smoothing_function = SmoothingFunction()

def bleu(x, y):
    if type(x) == str and type(y) == str:
        return sentence_bleu([x.split()], y.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function.method3)
    else:
        return 0.0  

dev["bleu_score"] = dev.apply(lambda x: bleu(x["model_response"], x["retrieved_response"]), axis=1)

print(f"Average BLEU Score: {dev['bleu_score'].mean()}")


# In[8]:


train_dev["user_prompt_clean"] = train_dev["user_prompt"].apply(clean_text)
test["user_prompt_clean"] = test["user_prompt"].apply(clean_text)

train_dev["tokenized"] = train_dev["user_prompt_clean"].apply(lambda x: x.split()) #tokenization
test["tokenized"] = test["user_prompt_clean"].apply(lambda x: x.split())

def sentence_embedding(tokens, model):
    word_vectors = [model[word] for word in tokens if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

train_dev["embedding"] = train_dev["tokenized"].apply(lambda tokens: sentence_embedding(tokens, fasttext_model))
test["embedding"] = test["tokenized"].apply(lambda tokens: sentence_embedding(tokens, fasttext_model))

train_dev_embeddings = np.vstack(train_dev["embedding"].values) #matrix form
test_embeddings = np.vstack(test["embedding"].values)

similarity_matrix = cosine_similarity(test_embeddings, train_dev_embeddings)
best_match_indices = similarity_matrix.argmax(axis=1)

retrieved_responses = train_dev["model_response"].iloc[best_match_indices].values
test["retrieved_response"] = retrieved_responses #saving the retrieved responses in the test dataset
test["response_id"] = train_dev.iloc[best_match_indices]["conversation_id"].values
submission = test[["conversation_id", "response_id"]]

submission.to_csv("track_2_test.csv", index=False)
print("Done")

