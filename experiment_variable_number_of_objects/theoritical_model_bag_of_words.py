

import os
import sys
sys.path.append(r'..')

from sentence_to_predicate import WordPredicate
import json
import sentence_grounding_test_parameters as param
import numpy as np
from recognized_object import RecognizedObject

from plots import *

import time
import matplotlib.pyplot as plt
import math

def caracteristics_to_output_teacher(object_lists, concept_to_output_id_dict,
                                     output_size, nb_concepts):
    """
        Transforms a list of caracteristics into a teacher numpy vector
    """

    targeted_output = np.zeros(output_size)

    for i, obj in enumerate(object_lists):

        offset = i * nb_concepts

        for j, concept in enumerate(obj):

            if concept is None:
                continue

            concept_id = offset + concept_to_output_id_dict[concept]
            targeted_output[concept_id] = 1.

    return targeted_output





def random_sentence_and_predicates():
    """
        Returns a ranodm sentence and predicates it contains.
        50% of the sentences concern one object only, 50% two.
    """

    nb_sentences = len(param.SENTENCE_TO_PREDICATE.items())
    rand_sentence_id_1 = np.random.choice(nb_sentences)
    sentence_1, predicate_1 = param.SENTENCE_TO_PREDICATE.items()[rand_sentence_id_1]

    if np.random.rand() < 0.5:
        rand_sentence_id_2 = np.random.choice(nb_sentences)
        sentence_2, predicate_2 = param.SENTENCE_TO_PREDICATE.items()[rand_sentence_id_2]
        sentence = sentence_1 + ' and ' + sentence_2
    else:
        sentence = sentence_1
        predicate_2 = None

    predicates = [predicate_1, predicate_2]
    return sentence, predicates

def random_recognized_object(category_choices = param.CATEGORIES,
                             position_choices = param.POSITIONS,
                             color_choices = param.COLORS):
    """
        Returns a random object. One half of the time,
        an empty object is returned.
    """

    if np.random.rand() < 0.5:
        random_category = np.random.choice(category_choices)
        random_position = np.random.choice(position_choices)
        random_color = np.random.choice(color_choices)
        return RecognizedObject(random_category, random_position, random_color)

    return RecognizedObject(None, None, None)


def random_object(category_choices = None,
                  position_choices = None,
                  color_choices = None):

    if category_choices is None:
        category_choices = param.CATEGORIES
        position_choices = param.POSITIONS
        color_choices = param.COLORS
    if np.random.rand() < 0.5:
        return [np.random.choice(category_choices), np.random.choice(position_choices), np.random.choice(color_choices)]
    else:
        return [None, None, None]

def possible_recognized_object_for_predicate(predicate, fill_unkown_fields = True):
    """
       From a predicate using words from sentence, returns an object from vision module
       corresponding to the situation described by the predicate (in french), using grounded concept
       (in english)
    """

    if predicate is None:

        if fill_unkown_fields:
            return random_recognized_object()

        return RecognizedObject(None, None, None)


    if fill_unkown_fields:
        default_category = np.random.choice(param.CATEGORIES)
        default_position = np.random.choice(param.POSITIONS)
        default_color = np.random.choice(param.COLORS)
    else:
        default_category = None
        default_position = None
        default_color = None

    seen_category = default_category
    seen_position = default_position
    seen_color = default_color

    seen_category = param.OBJ_NAME_TO_CONCEPT.get(predicate.object, default_category)
    seen_position = param.POSITION_NAME_TO_CONCEPT.get(predicate.action, default_position)
    seen_color = param.COLOR_NAME_TO_CONCEPT.get(predicate.color, default_color)

    return RecognizedObject(seen_category, seen_position, seen_color)



def possible_categories_for_predicate(predicate, fill_unkown_fields = True):
    """
       From a predicate using words from sentence, returns an object from vision module
       corresponding to the situation described by the predicate (in french), using grounded concept
       (in english)
    """
    if predicate is None:

        if fill_unkown_fields:
            return random_object()

        return [None, None, None]


    if fill_unkown_fields:
        default_category = np.random.choice(param.CATEGORIES)
        default_position = np.random.choice(param.POSITIONS)
        default_color = np.random.choice(param.COLORS)
    else:
        default_category = None
        default_position = None
        default_color = None


    seen_category = param.OBJ_NAME_TO_CONCEPT.get(predicate.object, default_category)
    seen_position = param.POSITION_NAME_TO_CONCEPT.get(predicate.action, default_position)
    seen_color = param.COLOR_NAME_TO_CONCEPT.get(predicate.color, default_color)

    return [seen_category, seen_position, seen_color]



def sentence_to_pred(sentence, sent_to_role):
    clauses = sentence.split(" and ")
    predicates = []
    if len(clauses) == 1:
        predicates.append(WordPredicate(clauses[0], sent_to_role[clauses[0]]))
        predicates.append(WordPredicate(None, None))
    else:
        for i in range(len(clauses)):
            predicates.append(WordPredicate(clauses[i], sent_to_role[clauses[i]]))
    return predicates

def softmax(x, beta = 1.):
    y = np.exp(beta * (x - np.max(x)))
    return y / np.sum(y)



def is_a_valid_imagined_object(predicate, imagined_object):
    """
        Returns whether the predicate description could apply to the imagine_object.

        Inputs:
            predicate: WordPredicate instance
            imagined_object: RecognizedObject instance

    """

    target = possible_recognized_object_for_predicate(predicate, fill_unkown_fields = False)

    for field in ['category', 'position', 'color']:

        wanted_field = getattr(target, field)

        if wanted_field is not None and getattr(imagined_object, field) != wanted_field:
            return False

    return True

def is_a_valid_representation(predicates, imagined_objects):
    """
        Returns whether each predicate  description could apply to its
        corresponding imagined_object.


        Inputs:
            predicates: a list of WordPredicate instances
            imagined_objects: a list of RecognizedObject instance
    """

    return all(map(is_a_valid_imagined_object, predicates, imagined_objects))




def is_an_exact_imagined_object(predicate, imagined_object):
    """
        Returns whether the imagined object matches exactly what the predicate
        describes.

        Inputs:
            predicate: WordPredicate instance
            imagined_object: RecognizedObject instance

    """
    target = possible_recognized_object_for_predicate(predicate, fill_unkown_fields = False)

    for field in ['category', 'position', 'color']:

        if getattr(imagined_object, field) != getattr(target, field):
            return False
    return True


def is_an_exact_representation(predicates, imagined_objects):
    """
        Returns whether the imagined object matches exactly what the predicate
        describes.

        Inputs:
            predicates: a list of WordPredicate instances
            imagined_objects: a list of RecognizedObject instance
    """

    return all(map(is_an_exact_imagined_object, predicates, imagined_objects))



def sentence_to_output_teacher_vector(sentence, sent_to_role, concept_to_output_id_dict, nb_concepts):
    ## sentences (str) -> list of predicate (WordPredicate) -> list of RecognizedObject -> list of categrories list -> input vector

    clauses = sentence.split(" and ")
    predicates = []
    if len(clauses) == 1:
        predicates.append(WordPredicate(clauses[0], sent_to_role[clauses[0]]))
        predicates.append(None)
    else:
        for i in range(len(clauses)):
            predicates.append(WordPredicate(clauses[i], sent_to_role[clauses[i]]))

    obj_list = list(map(possible_categories_for_predicate, predicates))
    #print("sentence : ", sentence, "obj list: ",obj_list)

    #the output size is 2*nb_cocpets because it can only be up to 2 objects per clause
    return caracteristics_to_output_teacher(obj_list, concept_to_output_id_dict, 2*nb_concepts, nb_concepts)

def output_to_vision(output, nb_concepts, factor):

    global concepts_delimitations
    global output_id_to_concept_dict
    objs = []
    nb_objects = 2
    for j in range(nb_objects):
        cat = [None] * 3
        offset = j*nb_concepts

        for i in range(len(concepts_delimitations)):
            cat_activations = output[offset+concepts_delimitations[i][0]:offset+concepts_delimitations[i][1]]
            #print(output_id_to_concept_dict[i], cat_activations, output, concepts_delimitations)


            proba = softmax(cat_activations)


            #print(proba)
            concept = np.argmax(proba)

            #the threshold to be accepted depend on the number of concept to choose from : must be a factor higher than the uniform proba
            threshold = factor / (concepts_delimitations[i][1] - concepts_delimitations[i][0])

            if proba[concept] < threshold:
                cat[i] = None
            else:
                cat[i] = output_id_to_concept_dict[concepts_delimitations[i][0] + concept]

        objs.append(RecognizedObject(cat[0], cat[1], cat[2])) #object, position, color

    return objs



## Bag of word model



def test_th_model_on_test_set(test_sentences, verbose):
    global nb_concepts, sent_to_role, concepts_delimitations, output_id_to_concept_dict

    test_outputs = []

    exact = 0
    valid = 0
    for i in range(len(test_sentences)):
        v = output_to_vision(bag_of_word_test(test_sentences[i]),nb_concepts, 1.3)
        pred = sentence_to_pred(test_sentences[i], sent_to_role)

        if is_an_exact_representation(pred, v):
            exact +=1

        if is_a_valid_representation(pred, v):
            valid +=1


    nb_sample = len(test_sentences)
    if verbose:
        print("Valid representations : ", valid,"/", nb_sample)
        print("Exact representations : ", exact,"/", nb_sample)


    return 1-valid/nb_sample, 1-exact/nb_sample

memory = {}




def filter_sentence(s):
    """Split s in two clauses and keep only key words"""
    sem_words = list(set(param.COLOR_NAMES + param.POSITION_NAMES + param.OBJECT_NAMES))

    clauses = s.split(" and ")
    clauses = [c.split(" ") for c in clauses]

    filtered_clauses = []
    for c in clauses:
        filt_clause = []
        for w in c:
            if w in sem_words:
                filt_clause.append(w)
        filt_clause.sort()
        filtered_clauses.append(str(filt_clause))
    return filtered_clauses


def bag_of_word_model_train(s, teacher_vect):
    """Keep the association bewteen the key words in s and the output in memory."""
    global memory, nb_concepts
    clauses = filter_sentence(s)
    for i in range(len(clauses)):
        memory[clauses[i]] = teacher_vect[i*nb_concepts: i*nb_concepts+nb_concepts]




def bag_of_word_test(s):
    """Return the output kept in memory if key words
       in s has been seen else a null vector"""
    global memory, nb_concepts
    f = filter_sentence(s)

    output = []

    for i in range(2):
        if i >= len(f):
            output.append(np.zeros(nb_concepts))
        else:
            c = f[i]
            if c in memory:
                output.append(memory[c])
            else:
                output.append(np.zeros(nb_concepts))
    return np.concatenate(output)


def train_and_test_bag_of_word(nb_object):
    global nb_concepts, sent_to_role, concepts_delimitations, output_id_to_concept_dict, memory, output_size

    #Prepropreccingand data generation
    param.create_dataset(nb_objects = nb_object)

    sent_to_role = param.SENTENCE_TO_ROLES
    sentences_one_object = list(sent_to_role.keys())
    sentences_two_objects = []


    concepts = param.CATEGORIES + param.POSITIONS + param.COLORS

    concepts_delimitations = [(0,len(param.CATEGORIES)),
                            (len(param.CATEGORIES), len(param.CATEGORIES) + len(param.POSITIONS)),
                            (len(param.CATEGORIES) + len(param.POSITIONS), len(param.CATEGORIES) + len(param.POSITIONS)+ len(param.COLORS))]

    nb_concepts = len(concepts)

    output_size = 2*nb_concepts


    concept_to_output_id_dict = {}
    output_id_to_concept_dict = {}
    for i,c in enumerate(concepts):
        concept_to_output_id_dict[c] = i
        output_id_to_concept_dict[i] = c

    np.random.shuffle(sentences_one_object)

    train_one_obj = 300
    train_two_objs = 700


    test_one_obj = 300
    test_two_objs = 700


    valid_ers = []
    exact_ers = []

    for j in range(10):
        memory = {}


        train_sentences = set(sentences_one_object[:train_one_obj])

        for i in range(train_two_objs):
            s1 = np.random.choice(sentences_one_object)
            s2 = np.random.choice(sentences_one_object)
            train_sentences.add( s1 + " and "+ s2 )


        test_sentences = set(sentences_one_object[-test_one_obj:])
        for i in range(test_two_objs):
            s1 = np.random.choice(sentences_one_object)
            s2 = np.random.choice(sentences_one_object)
            s = s1 + " and "+ s2
            if not(s in train_sentences):
                test_sentences.add(s)

        test_sentences = list(test_sentences)
        train_sentences = list(train_sentences)

        if j ==5:
            print("Number of possible sentences:", len(sentences_one_object)**2 + len(sentences_one_object))

        trainY = np.array([sentence_to_output_teacher_vector(s, sent_to_role, concept_to_output_id_dict, nb_concepts) for s in train_sentences])

        testY = np.array([sentence_to_output_teacher_vector(s, sent_to_role, concept_to_output_id_dict, nb_concepts) for s in test_sentences])

        ##th model training

        for i in range(len(train_sentences)):
            bag_of_word_model_train(train_sentences[i], trainY[i])


        ##th model testing

        valid_er, exact_er = test_th_model_on_test_set(test_sentences, False)
        exact_ers.append(exact_er)
        valid_ers.append(valid_er)
    return np.mean(valid_ers), np.mean(exact_ers)


with open("mean_perf_bag_of_words.txt", "a") as f:
    f.write("Nb of objects, Valid error, Exact error\n")


#test the bag of word model for several number of objects
for nb in range(3,51):
    print(nb, "objects:")
    v, e = train_and_test_bag_of_word(nb)
    print("Valid error:", np.round(v,4))
    print("Exact error:", np.round(e,4))


    with open("mean_perf_bag_of_words.txt", "a") as f:
       f.write(str(nb) + ","+str(v) + "," + str(e) + "\n")
    print("-----------------------------------")






