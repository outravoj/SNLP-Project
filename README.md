# SNLP-Project
This is a natural language processing project created for Aalto course SNLP. All the code was developed in Google Colab and this repository serves a storage for the final version of the project.

##Files
In this section I will comment on purpose of each file in this repository
* Exploratory data analysis.ipynb - Exploration of given dataset.
* Basic_BERT_comparison.ipynb - In this notebook we trained a BERT model for sequence classification with custom linear layer.
* Electra_out_of_the_box.ipynb - This notebook contrains training of out of the box SELECTRA model for sequence classification.
* SElectra.ipynb - Training of the sentence ELECTRA model which outputs embedding for a given sentence. The model was trained also on the given dataset.
* SBert.ipynp - This notebook was meant for trainig sentence BERT model, but due to lack of sufficient computational resources we could not train this model.
* Selectra.py - definiton of SELECTRA class (architecture)
* Sbert.py - definition of SBERT class (architecture)
* LSH - * Gather_save_sentences.ipynb - indexing of sentences into a dictionary used for training and querying LSH model.
        * LSH_train.ipynb - Trainig of LSH model used to approximate nearest neigbourhood for sentence architectures outputting sentence embeddings
        * LSH_eval.ipynb - Evaluation of LSH model (try out few sentences)
        * best_model_state.bin - Best trained SELECTRA model
        * lsh_model.pickle - Saved LSH model for querying
        * sentences_dict.json - The indexed sentences from Gather_save_sentences.ipynb
