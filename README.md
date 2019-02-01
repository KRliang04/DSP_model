# DSP_model

This repository is for predicting the user satisfaction of the chatbot.

### Dataset
We used and adapted open source data -- DSTC 2 which contains a number of dialogs related to restaurant search. You can find it [here](http://camdial.org/~mh521/dstc/)


Before running the code please download the google word2vec model [download](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)

### Training the model:

#### python feature_eng.py 1

There are following files will be generated:

* tfidf.dict, tfidf.sav 

* features.csv 

* LR_model.sav

### Predicting the result

#### python feature_eng.py 2 "asolute path to the text file"







