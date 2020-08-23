from sentence_to_predicate import WordPredicate
import json
import sentence_grounding_test_parameters as param
import numpy as np
from recognized_object import RecognizedObject

from plots import *

## Data generation



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


def sentence_to_output_teacher_vector(sentence, sent_to_role, concept_to_output_id_dict, nb_concepts):
    """"
    Pipeline :
    sentences (str) -> list of predicate (WordPredicate) -> list of RecognizedObject -> list of categrories list -> input vector
    """

    clauses = sentence.split(" and ")
    predicates = []
    if len(clauses) == 1:
        predicates.append(WordPredicate(clauses[0], sent_to_role[clauses[0]]))
        predicates.append(None)
    else:
        for i in range(len(clauses)):
            predicates.append(WordPredicate(clauses[i], sent_to_role[clauses[i]]))

    obj_list = list(map(possible_categories_for_predicate, predicates))
    #print("obj list:",obj_list)

    #the output size is 2*nb_cocpets because it can only be up to 2 objects per clause
    return caracteristics_to_output_teacher(obj_list, concept_to_output_id_dict, 2*nb_concepts, nb_concepts)


### handle of word dictionnary and one hot encoding

word2one_hot_id = {}
nb_unique_words = 0
def init_one_hot_encoding(sentences):
    global word2one_hot_id
    global nb_unique_words
    ind = 0
    for s in sentences:
        for w in s.split(" "):

            if not(w in word2one_hot_id):
                word2one_hot_id[w] = ind
                ind +=1
    nb_unique_words = len(word2one_hot_id)


def one_hot_encoding(word):
    global word2one_hot_id
    vect = np.zeros(nb_unique_words)
    vect[word2one_hot_id[word]] = 1.
    return vect

def one_hot_encoding_sentence(sentence):
    if sentence == "":
        return None
    return np.array(list(map(one_hot_encoding, sentence.split(" "))))


## representation handling

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


def output_to_vision(output, nb_concepts, factor,
                     concepts_delimitations, output_id_to_concept_dict):
    objs = []
    nb_objects = 2
    for j in range(nb_objects):
        cat = [None] * 3
        offset = j*nb_concepts

        for i in range(len(concepts_delimitations)):
            cat_activations = output[offset+concepts_delimitations[i][0]:offset+concepts_delimitations[i][1]]
            proba = softmax(cat_activations)
            concept = np.argmax(proba)

            #the threshold to be accepted depend on the number of concept to choose from : must be a factor higher than the uniform proba
            threshold = factor / (concepts_delimitations[i][1] - concepts_delimitations[i][0])

            if proba[concept] < threshold:
                cat[i] = None
            else:
                cat[i] = output_id_to_concept_dict[concepts_delimitations[i][0] + concept]

        objs.append(RecognizedObject(cat[0], cat[1], cat[2])) #object, position, color

    return objs

def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""

    # Set the seed
    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print( "!!! WARNING !!!: Seed was not set correctly.")
        print( "!!! Seed that we tried to use: "+str(seed))
        print( "!!! Error message: "+str(e))
        seed = None
    #print( "Seed used for random values:", seed)
    return seed

def sigmoid(x, k=1.0):
    return 1/(1+ np.exp(-k*x))