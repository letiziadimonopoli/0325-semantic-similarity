# Semantic Similarity Measurement (March 2025)

🎯 Goal

The goal of this project is to implement a retrieval method that can identify the most relevant response from a dataset given a new question.

🛠 Methodology  

Track 1: Discrete Text Representation
- Preprocessing: lowering cases and removing leading and trailing whitespaces.
- Methodology: TF-IDF.

Track 2: Distributed Static Text Representation
- Preprocessing: lowering cases, fixing contractions, removing punctuation, removing extra spaces, lemmatization and tokenization.
- Methodology: FastText model.

Track 3: Open Text Representation
- Methodology: all-mpnet-base-v2 model.
