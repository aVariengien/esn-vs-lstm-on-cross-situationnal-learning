# esn-vs-lstm-on-cross-situationnal-learning


#### Context of creation
This repository compile the work I did during an internship in 2020 in the Mnemosyne Team (Inria Bordeaux) with Xavier Hinaut.

## Content

This work includes: 
* A comparison between Long-Short Term Memory networks (LSTM) and Echo States Networks (ESN) on a Cross-Situationnal Learning task (CSL).
* Visualisation of the activities of LSTM cells and ESN reservoir units.
* Visualisation using the dimensionnal reduction technique UMAP of the structure of recurrent states space for both LSTM and ESN.

## Files

 * Code developped by Alexis Juven (@aJuvenn) from the repository https://github.com/aJuvenn/JuvenHinaut2020_IJCNN
    * `sentence_to_predicate.py`
    * `sentence_grounding_test_parameters.py` - to create the sentences according to a context free grammar.
    * `recognized_object.py`
    * `predicate_manipulation.py`
    * `grammar_manipulation.py`
 * Code in part developped by Dinh Thanh Trung and me 
    * `data_processing.py` - useful function to create proper data sets for model training
 * Files to be executed
    * `CSL_ESN.py`- Working implementation of an ESN performing the CSL task. Can be ran from the terminal with the arguments `python CSL_ESN.py *nb_objects (int)* *minimal mode {0 or 1}* `
    * `CSL_LSTM.py` - Working implementation of a LSTM performing the CSL task and qualitative analysis of the activity of LSTM cells. Can also be run from the terminal with the same synthax.
    * `reservoir_activity_analysis.py` - Visualisation of the activity of reservoir units.
 * Folder `RSSviz`
    * `RSSviz.py` - tools using the `umap` module to perform the dimensionnal reduction.
    * `RSSviz_ESN.py` and `RSSviz_LSTM.py` - Recurent State Space Visualisations from both models.
  * Folder `experiment_variable_number_of_objects`
    * `xp_variable_nb_objects_ESN.sh` and `xp_variable_nb_objects_LSTM.sh` - `bash` program to evaluate performance of the models on data sets with different number of objects.
    * `theoritical_model_bag_of_words.py` - implementation of the theoritical model based on an interpretation of sentences as bag of words.
    
## Requirements
The python modules needed to run the code.
* For the ESN:
```
cycler==0.10.0
dill==0.3.2
joblib==0.16.0
kiwisolver==1.2.0
matplotlib==3.2.2
numpy==1.19.0
pkg-resources==0.0.0
pyparsing==2.4.7
python-dateutil==2.8.1
reservoirpy==0.2.0
scipy==1.5.1
six==1.15.0
tqdm==4.47.0
```

* For the LSTM:
```
absl-py==0.9.0
astunparse==1.6.3
cachetools==4.1.1
certifi==2020.6.20
chardet==3.0.4
cycler==0.10.0
gast==0.3.3
google-auth==1.18.0
google-auth-oauthlib==0.4.1
google-pasta==0.2.0
grpcio==1.30.0
h5py==2.10.0
idna==2.10
importlib-metadata==1.7.0
Keras==2.4.3
Keras-Preprocessing==1.1.2
kiwisolver==1.2.0
Markdown==3.2.2
matplotlib==3.2.2
numpy==1.19.0
oauthlib==3.1.0
opt-einsum==3.2.1
protobuf==3.12.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyparsing==2.4.7
python-dateutil==2.8.1
PyYAML==5.3.1
requests==2.24.0
requests-oauthlib==1.3.0
rsa==4.6
scipy==1.4.1
six==1.15.0
tensorboard==2.2.2
tensorboard-plugin-wit==1.7.0
tensorflow==2.2.0
tensorflow-estimator==2.2.0
termcolor==1.1.0
tqdm==4.47.0
urllib3==1.25.9
Werkzeug==1.0.1
wrapt==1.12.1
zipp==3.1.0
```
