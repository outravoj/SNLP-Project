# SNLP-Project
This is a natural language processing project created for Aalto course SNLP. All the code was developed in Google Colab and this repository serves a storage for the final version of the project.

## Dataset 
Our dataset used tot rain our models was the kaggle dataset https://www.kaggle.com/c/quora-question-pairs. This model consists of question pairs along with label that indicates whether the questions are duplicates of each other. The dataset is saved in data/train.csv.

## Files
In this section I will comment on purpose of each file in this repository
* Exploratory data analysis.ipynb - Exploration of given dataset.
* Basic_BERT_comparison.ipynb - In this notebook we trained a BERT model for sequence classification with custom linear layer.
* Electra_out_of_the_box.ipynb - This notebook contrains training of out of the box SELECTRA model for sequence classification.
* SElectra.ipynb - Training of the sentence ELECTRA model which outputs embedding for a given sentence. The model was trained also on the given dataset.
* SBert.ipynp - This notebook was meant for trainig sentence BERT model, but due to lack of sufficient computational resources we could not train this model.
* Selectra.py - definiton of SELECTRA class (architecture)
* Sbert.py - definition of SBERT class (architecture)
* LSH/Gather_save_sentences.ipynb - indexing of sentences into a dictionary used for training and querying LSH model.
* LSH/LSH_train.ipynb - Trainig of LSH model used to approximate nearest neigbourhood for sentence architectures outputting sentence embeddings
* LSH/LSH_eval.ipynb - Evaluation of LSH model (try out few sentences)
* LSH/best_model_state.bin - Best trained SELECTRA model
* LSH/lsh_model.pickle - Saved LSH model for querying
* LSH/sentences_dict.json - The indexed sentences from Gather_save_sentences.ipynb

## Running the code
If you want try out the trained LSH model with the best SELECTRA, you can download this folder but you have to change the paths present in the file. As this project was done in Google Colab, you have to change the path in the following manner. For every path, which consists of: '/content/gdrive/MyDrive/Colab Notebooks/SNLP projekt', you have to change just the path preceding 'SNLP projekt'.
