
import sys
from sentence_to_predicate import WordPredicate
import json
import sentence_grounding_test_parameters as param
import numpy as np
from recognized_object import RecognizedObject

from plots import *
from data_processing import *

import time
import matplotlib.pyplot as plt
import math
from reservoirpy import ESNOnline, mat_gen, ESN
from reservoirpy.mat_gen import generate_internal_weights, generate_input_weights



def test_with_sentences_ESN(sentences, model, nb_concepts, threshold_factor):
    test = [one_hot_encoding_sentence(s) for s in sentences]
    res = []
    model.reset_reservoir()
    for i in range(len(sentences)):
        outputs, int_states = model.run([test[i],])
        res.append(outputs[0])
        model.reset_reservoir()
    return res, [output_to_vision(res[j][-1],nb_concepts, threshold_factor, concepts_delimitations, output_id_to_concept_dict) for j in range(len(res))]




def test_on_test_set(model, test_sentences, testX, testY, verbose, threshold_factor, online):

    test_outputs = []
    rmse = 0
    for sent_nb in range(len(testX)):
        if online:
            model.reset_reservoir()
        outputs, int_states = model.run([testX[sent_nb],])
        test_outputs.append(outputs[0][-1])
        rmse += np.mean( (outputs[0][-1] - testY[sent_nb])**2)
    rmse = np.sqrt(rmse/len(testX))

    if verbose:
        print("End of testing")
    exact = 0
    valid = 0
    for i in range(len(test_outputs)):
        v = output_to_vision(test_outputs[i],nb_concepts, threshold_factor, concepts_delimitations, output_id_to_concept_dict)
        pred = sentence_to_pred(test_sentences[i], sent_to_role)


        if is_an_exact_representation(pred, v):
            exact +=1

        if is_a_valid_representation(pred, v):
            valid +=1

        if is_a_valid_representation(pred, v) and not(is_an_exact_representation(pred, v)):
            pass

    nb_sample = len(testX)
    if verbose:
        print("Valid representations : ", valid,"/", nb_sample)
        print("Exact representations : ", exact, "/", nb_sample)
    return 1-valid/nb_sample, 1-exact/nb_sample, rmse


def get_int_states(model, sentences, one_chunck = True, only_last = False, get_obj_nb = False):

    test = [one_hot_encoding_sentence(s) for s in sentences]
    int_states_on_sent = []
    labels = []

    clause_nb = []

    for i in range(len(sentences)):
        model.reset_reservoir()
        outputs, int_states = model.run([test[i],])

        if only_last:
            int_states_on_sent.append(int_states[0][-1].reshape((int_states[0][-1].shape[0])))
        else:
            if one_chunck:
                for k in range(int_states[0].shape[0]):
                    int_states_on_sent.append(int_states[0][k].reshape((int_states[0][k].shape[0])))
            else:
                int_states_on_sent.append(int_states[0].reshape((int_states[0].shape[0], int_states[0].shape[1])))

            if get_obj_nb:
                words = sentences[i].split(" ")
                clause_id = 0
                for w in range(len(words)):
                    if words[w] == "and":
                        clause_id +=1
                    clause_nb.append(clause_id)
                    if words[w] == "and":
                        clause_id +=1
                clause_nb[-1] = 3 #final states

        words = sentences[i].split(" ")
        for j in range(1, len(words)+1):
            labels.append(" ".join(words[:j]))

    if one_chunck:
        int_states_on_sent = np.array(int_states_on_sent)

    if get_obj_nb:
        return int_states_on_sent, labels, np.array(clause_nb)
    else:
        return int_states_on_sent, labels

## functions used to create RSSviz showing the position of the errors
def getErrorScore(pred, obj):
    if is_an_exact_imagined_object(pred, obj):
        return 2
    elif is_a_valid_imagined_object(pred, obj):
        return 1
    else:
        return 0

def errorCode(obj_er):
    return (obj_er[0] + 3*obj_er[1])


def getErrorsInfo(reservoir_state, sentences, Wout, threshold_fact = 1.3):
    global nb_concepts, sent_to_role
    errors = []
    ind = 0
    for s in sentences:
        pred = sentence_to_pred(s, sent_to_role)
        for i in range(len(s.split(" "))):
            output = np.dot(Wout, reservoir_state[ind])
            v = output_to_vision(output,nb_concepts, threshold_fact,
                                 concepts_delimitations, output_id_to_concept_dict)
            obj_errors = (getErrorScore(pred[0], v[0]), getErrorScore(pred[1], v[1]))
            errors.append(errorCode(obj_errors))
            ind +=1
    return np.array(errors)

################### MAIN #####################

## Parameters


add_begin_end = True #add the word "BEGIN" at the beggining and "END" at the end of all sentences ?
verbose_training = True
continuous_sentence_training = False # continuous or final training


use_save = False # use saved matrices from previous run ?
if use_save:
    name_id = '0.8885270787313605' #example random id of the files to load
    pth = r"saved_ESN/" #path to the saved arrays

threshold_factor = 1.3 # the factor used to get the threshold in the creation of he discrete representation
nb_objets = 4 #number of objects in the vocabulary used to generate the sentences (the bigger the harder)
N = 1000 #the size of the reservoir


#this code can be used in command line to specify the number of object and easily test different number of objects

minimal_mode = False #if minimal mode is on, the ESN will only be trained and tested on test set. The only text print will be "nb of objects, valid error on test set, exact error on test set, RMSE on test set, time to train"

## Handling the arguments when the script is ran in a shell

if len(sys.argv) > 1:
    if "-h" in sys.argv:
        print("Argument: number of objects (int), minimal mode (int : 0 or 1)")
        quit()
    if len(sys.argv) != 3:
        print("Argument: number of objects (int), minimal mode (int : 0 or 1)")
        raise ValueError("Invalid argument.")
    try:
        nb_objects = int(sys.argv[1])
        minimal_mode_int = int(sys.argv[2])
        if not(minimal_mode_int in [0,1]):
            print("Argument: number of objects (int), minimal mode (int : 0 or 1)")
            raise ValueError("Invalid argument.")
        minimal_mode = bool(minimal_mode_int)
        verbose_training = verbose_training and not(minimal_mode)
    except ValueError:
        print("Argument: number of objects (int), minimal mode (int : 0 or 1)")
        raise ValueError("Invalid argument.")

##dataset initialisation : creation of the sentence according to the grammar
param.create_dataset(nb_objects = nb_objets)


###generate train and test sets


#one hot encoding initialisation
sent_to_role= param.SENTENCE_TO_ROLES

other_words = ['and']

if add_begin_end:
    other_words.append("BEGIN")
    other_words.append("END")

init_one_hot_encoding(list(sent_to_role.keys()) + other_words)
nb_unique_words = len(word2one_hot_id)

#concept dictionnary initialisation (it's the link between the output position and their meaning)

concepts = param.CATEGORIES + param.POSITIONS + param.COLORS

concepts_delimitations = [(0,len(param.CATEGORIES)),
                          (len(param.CATEGORIES),
                          len(param.CATEGORIES) + len(param.POSITIONS)),
                          (len(param.CATEGORIES) + len(param.POSITIONS),
                          len(param.CATEGORIES) + len(param.POSITIONS)+ len(param.COLORS))]

nb_concepts = len(concepts)

output_size = 2*nb_concepts

concept_to_output_id_dict = {}
output_id_to_concept_dict = {}
for i,c in enumerate(concepts):
    concept_to_output_id_dict[c] = i
    output_id_to_concept_dict[i] = c


##generate data


sentences_one_object = list(sent_to_role.keys())
sentences_two_objects = []

for s1 in sentences_one_object:
    for s2 in sentences_one_object:
        sentences_two_objects.append(s1 + " and " + s2)


#we adjust the different dictionnaries to include sentences with BEGIN and END
if add_begin_end:
    for i in range(len(sentences_one_object)):

        sent_to_role["BEGIN "+ sentences_one_object[i]+ " END"] = [0] + sent_to_role[sentences_one_object[i]] + [0]
        sent_to_role["BEGIN "+ sentences_one_object[i]] = [0] + sent_to_role[sentences_one_object[i]]
        sent_to_role[ sentences_one_object[i] + " END"] = sent_to_role[sentences_one_object[i]] + [0]
        sentences_one_object[i] = "BEGIN "+ sentences_one_object[i]+ " END"

    for i in range(len(sentences_two_objects)):
        sentences_two_objects[i] = "BEGIN "+ sentences_two_objects[i]+ " END"



np.random.shuffle(sentences_one_object)
np.random.shuffle(sentences_two_objects)

train_one_obj = 300
train_two_objs = 700


test_one_obj = 300
test_two_objs = 700

train_sentences = (sentences_one_object[:train_one_obj]
              + sentences_two_objects[:train_two_objs])


test_sentences = (sentences_one_object[-test_one_obj:]
              + sentences_two_objects[-test_two_objs:])



trainX = [one_hot_encoding_sentence(s) for s in train_sentences]
trainY = np.array([sentence_to_output_teacher_vector(s,
                                                     sent_to_role,
                                                     concept_to_output_id_dict,
                                                     nb_concepts) for s in train_sentences])


testX = [one_hot_encoding_sentence(s) for s in test_sentences]
testY = np.array([sentence_to_output_teacher_vector(s,
                                                    sent_to_role,
                                                    concept_to_output_id_dict,
                                                    nb_concepts) for s in test_sentences])





## create the ESN
iss = 1
nb_features = nb_unique_words
set_seed(None) #we test on a different and random seed each time

if continuous_sentence_training: # Alexis Juven's hyper-parameters optimized through random search
    sr = 1.3
    sparsity = 0.81
    leak = 0.04
    alpha_coef = 10.**(-3.7)
else:
    sr = 1.1
    sparsity = 0.85
    leak = 0.05
    alpha_coef = 10.**(-3.5)



# build an ESN online, i.e. trained with FORCE learning after each sample

W = mat_gen.fast_spectra_initialization(N, spectral_radius=sr, proba = sparsity) #reservoir matrix
Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=nb_features, #input matrix
                                    input_bias=True, input_scaling=iss)
Wout = np.zeros((output_size, N+1)) #output matrix to be optimized


reservoir = ESNOnline(lr = leak,
                    W = W,
                    Win = Win,
                    Wout = Wout,
                    alpha_coef = alpha_coef)



if __name__ == "__main__":

    if use_save:
        Wout = np.load(pth+"Wout"+name_id+".npy", allow_pickle = True)
        Win = np.load(pth+"Win"+name_id+".npy", allow_pickle = True)
        W = np.load(pth+"W"+name_id+".npy", allow_pickle = True)
        W = W.item()
        reservoir = ESNOnline(lr = leak,
                            W = W,
                            Win = Win,
                            Wout = Wout,
                            alpha_coef = alpha_coef)

    if not(use_save):
        t1 = time.process_time()

        for sent_nb in range(len(trainX)): #training sentences

            if continuous_sentence_training:
                wash_init_steps = 0 #we train at each step
            else:
                wash_init_steps = trainX[sent_nb].shape[0]-1 #we train only the last step


            reservoir.reset_reservoir()
            reservoir.train(inputs=np.array([trainX[sent_nb]]),
                            teachers=np.array([[trainY[sent_nb]]*trainX[sent_nb].shape[0]]), #the teacher is always the same vector
                            wash_nr_time_step=wash_init_steps,
                            verbose=False)


            if sent_nb%100 == 0 and verbose_training:
                print("Advancement :")
                print( sent_nb/len(trainX))


        t2 = time.process_time()
        if verbose_training:
            print("CPU Time to train : ", t2 - t1, " s")

        ##saving
        if not(minimal_mode):
            print("Saving the matrices ...")
            rd_id = np.random.random()
            print("ID for file saved : "+ str(rd_id))
            np.save(r"saved_ESN/Wout"+str(rd_id), Wout)
            np.save(r"saved_ESN/Win"+str(rd_id), Win)
            np.save(r"saved_ESN/W"+str(rd_id), W)
            print("Matrices saved !")

        ##Testing
        if verbose_training:
            print("Testing on test set...")
        vtest, extest, rmsetest = test_on_test_set(reservoir, test_sentences, testX, testY, verbose_training, threshold_factor, True)

        if minimal_mode:
            print(str(nb_objects) + "," +str(vtest) + "," + str(extest) + ","+ str(rmsetest) + "," + str(t2-t1))

        if not(minimal_mode):
            print("Testing on train set...") #to compare for overfitting estimation
            vtr, extr, rmsetr = test_on_test_set(reservoir, train_sentences, trainX, trainY, True, threshold_factor, True)



    if not(minimal_mode):
        ##Qualitative testing

        #sample sentences for model analysis
        test_sent = ["BEGIN on the middle is a green glass and that is a orange bowl END", #0
        "BEGIN on the middle is a green glass and that is a orange bowl END",
        "BEGIN a blue bowl is on the right and on the right there is the blue bowl END",
        "BEGIN on the middle there is the orange and the orange on the middle is orange END", #3
        "BEGIN the orange is orange and on the middle there is the green cup END",
        "BEGIN on the left there is a blue glass and this is a orange END",
        "BEGIN the glass on the left is blue and the bowl on the right is green END", #6
        "BEGIN the orange on the right is green and there is a blue cup on the middle END",
        "BEGIN on the right there is a red glass and the cup on the middle is red END",
        "BEGIN a green glass is on the right and on the left is a orange END"] #9


        outputs, vision = test_with_sentences_ESN(test_sent, reservoir, nb_concepts, threshold_factor)

        #plot final outputs on the first test sentence
        id = 0
        plot_final_activation(outputs[id][-1], concepts_delimitations, output_id_to_concept_dict, nb_concepts, test_sent[id])


        #plot and save to png the evolution of the outputs during the processing of the first test sentence (plots developped by Alexis Juven)


        id = 0
        s = test_sent[id]

        if continuous_sentence_training:
            output_fun = lambda x: x
        else:
            output_fun = sigmoid #sigmoid function is recommended when dealing with final learning

        plot_concept_activation(s,
                                outputs[id],
                                concepts_delimitations,
                                nb_concepts,
                                savefig = True,
                                sub_ttl_fig = s+ " ESN",
                                output_function = output_fun)


        ##reservoir activity visualisation
        res_states, vis = test_with_sentences_ESN(test_sentences, reservoir, nb_concepts, 1)
        s = test_sentences[500]

        res_act , _ = get_int_states(reservoir, [s], one_chunck = False) #get reservoir states

        plot_hidden_state_activation(s,
                                    [res_act[0]],
                                    state='reservoir',
                                    units_to_plot = [i for i in range(20)], #plot the activation of the 20 first reseroir units (arbitrary choice)
                                    plot_variation = False,
                                    plot_sum = False)


