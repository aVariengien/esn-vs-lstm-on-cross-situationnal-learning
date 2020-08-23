#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from grammar_manipulation import role_for_words, cat, union, maybe, sentence_to_roles
from predicate_manipulation import WordPredicate, NO_ROLE, ACTION, OBJECT, COLOR

from collections import defaultdict
import numpy as np
import tqdm





def create_dataset(high_difficulty = True, nb_objects = 3):
    
    # object_names = ['cup', 'orange', 'bowl', 'apple', 'spoon']
    
    #object_names = ['cup', 'bowl', 'apple', 'spoon'] # remove orange to avoid polysemous words for now
    # color_names = ['red', 'orange', 'green', 'blue']
    #color_names = ['red', 'orange', 'yellow', 'green', 'blue', 'magenta']
    
    position_names = ['left', 'middle', 'right']

    object_names = ['glass', 'cup', 'bowl', 'orange','spoon', 
                    'apple', 'accordion', 'acoustic','bagpipes',
                    'banjo','bass','bongo','bugle',
                    'cello','clarinet','cymbals','drums',
                    'electric','guitar','flute','horn',
                    'harmonica','harp','keyboard','maracas',
                    'organ','pan','piano','recorder',
                    'saxophone','sitar','tambourine','triangle',
                    'trombone','trumpet','tuba','ukulele',
                    'violin','xylophone','bassoon','castanets',
                    'didgeridoo','double','gong','harpsichord',
                    'lute','mandolin','oboe','piccolo','viola']

    object_names = object_names[:nb_objects]
    color_names = ['red', 'orange', 'green', 'blue']
    position_names = ['left', 'right', 'middle']
     
    clean_dataset()
    
    for obj in object_names:
        add_object(obj, build_after = False)
        
    for col in color_names:
        add_color(col, build_after = False)
        
    for pos in position_names:
        add_position(pos, build_after = False)
    
    # add_position('center', '<middle_pos>', build_after = False)
    
    build_all()


def clean_dataset():
    
    global OBJECT_NAMES, COLOR_NAMES, POSITION_NAMES
    global CATEGORIES, POSITIONS, COLORS
    global OBJ_NAME_TO_CONCEPT, COLOR_NAME_TO_CONCEPT, POSITION_NAME_TO_CONCEPT
    
    OBJECT_NAMES = []
    COLOR_NAMES = []
    POSITION_NAMES = []
    CATEGORIES = []
    POSITIONS = []
    COLORS = []
    OBJ_NAME_TO_CONCEPT = dict()
    COLOR_NAME_TO_CONCEPT = dict()
    POSITION_NAME_TO_CONCEPT = dict()


def build_all():
    
    global SENTENCE_TO_ROLES
    global SENTENCE_TO_PREDICATE
    global CONCEPT_LISTS
    global VISION_ENCODER

    CONCEPT_LISTS = [
            CATEGORIES,
            POSITIONS,
            COLORS
            ]

    SENTENCE_TO_ROLES = create_grammar()
    SENTENCE_TO_PREDICATE = {s : WordPredicate(s, r) for s, r in SENTENCE_TO_ROLES.items()}


def create_grammar():

    OBJ = role_for_words(OBJECT, OBJECT_NAMES)
    COL = role_for_words(COLOR, COLOR_NAMES)
    
    POSITIONS = role_for_words(ACTION, POSITION_NAMES)
    
    
    IS_ACTION = role_for_words(ACTION, ['is'])
    IS_NOROLE = role_for_words(NO_ROLE, ['is']) 
    
    TO_THE = union(
            sentence_to_roles('on the', [NO_ROLE, NO_ROLE])
    )
    
    THIS_IS = union(
            sentence_to_roles('this is', [ACTION, NO_ROLE]),
            sentence_to_roles('that is', [ACTION, NO_ROLE])      
    ) # 2
    
    THERE_IS = sentence_to_roles('there is', [NO_ROLE, NO_ROLE])
    
    DET = role_for_words(NO_ROLE, ['a', 'the']) # 2
    
    GN = cat(DET, maybe(COL), OBJ) # 70 -> 10 none col
    
    TO_THE_POSITION = cat(TO_THE, POSITIONS) # 3
    
    return union(
        cat(THIS_IS, GN), # 140 -> 20 none both + 120 none pos
        cat(DET, OBJ, IS_ACTION, COL), # 2x5x6 = 60 -> 60 none pos
        cat(DET, OBJ, TO_THE_POSITION, IS_NOROLE, COL), # 2x5x3x6 = 180
        cat(GN, IS_NOROLE, TO_THE_POSITION), # 70x3=210 -> 30 none col
        cat(THERE_IS, GN, TO_THE_POSITION), # 210 -> 30 none col
        cat(TO_THE_POSITION, union(IS_NOROLE, THERE_IS), GN) # 420 -> 60 none col
    )
    # Different none col predicates = 5obj x 3pos
    # Different none pos predicates = 5obj x 6col x 3(this_is, that_is, obj_is_col)
    # Different none both predicates = 5obj x 2(this_is, that_is)


def add_object(name, concept = None, build_after = True):
    
    if concept is None:
        concept = '<' + name.lower() + '_obj>'
    
    if name not in OBJECT_NAMES:
        OBJECT_NAMES.append(name)
        
    if concept not in CATEGORIES:
        CATEGORIES.append(concept)
        
    OBJ_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()
    
    
def add_position(name, concept = None, build_after=True):
    
    if concept is None:
        concept = '<' + name.lower() + '_pos>'
    
    if name not in POSITION_NAMES:
        POSITION_NAMES.append(name)
    
    if concept not in POSITIONS:
        POSITIONS.append(concept)
        
    POSITION_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()
       
        
def add_color(name, concept = None, build_after=True):
    
    if concept is None:
        concept = '<' + name.lower() + '_col>'
    
    
    if name not in COLOR_NAMES:
        COLOR_NAMES.append(name)
    
    if concept not in COLORS:
        COLORS.append(concept)
        
    COLOR_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()


    

def possible_complete_predicates():
    possible_predicates = set()
    for pos in POSITION_NAMES:
        for obj in OBJECT_NAMES:
            for col in COLOR_NAMES:
                possible_predicates.add(pos+"("+obj+","+col+")")
    return possible_predicates


def decompose_predicate_components(predicate):
    
    # Transform predicate to string and strip all spaces
    pred = str(predicate).lower().replace(' ','')
    if pred == 'none':
        return None, None, None
    
    # Parse the string to category, color and prediction
    position = pred[:pred.find('(')]
    if pred.find(',') != -1:
        color = pred[pred.find(',')+1:pred.find(')')]
        if color not in COLOR_NAMES:
            color = None
        category = pred[pred.find('(')+1:pred.find(',')]
    else:
        color = None
        category = pred[pred.find('(')+1:pred.find(')')]
        
    return category, position, color


def decompose_predicate(predicate):
    
    # Transform predicate to string and strip all spaces
    pred = str(predicate).lower().replace(' ','')
    if pred == 'none':
        return None, None, None, False
    
    # Parse the string to category, color and prediction
    is_complete_predicate = True
    position = pred[:pred.find('(')]
    if position not in POSITION_NAMES:
        position = None
        is_complete_predicate = False
    if pred.find(',') != -1:
        color = pred[pred.find(',')+1:pred.find(')')]
        if color not in COLOR_NAMES:
            color = None
        category = pred[pred.find('(')+1:pred.find(',')]
    else:
        color = None
        category = pred[pred.find('(')+1:pred.find(')')]
        is_complete_predicate = False
    return category, position, color, is_complete_predicate


def is_complete_predicate(predicate):
    _, _, _, is_complete = decompose_predicate(predicate)
    return is_complete 


def exact_same_predicates(predicate1, predicate2):
    cat1, pos1, col1, _ = decompose_predicate(predicate1)
    cat2, pos2, col2, _ = decompose_predicate(predicate2)
    return cat1 == cat2 and pos1 == pos2 and col1 == col2    

def similar_color_or_position(att1, att2, ATT_NAMES):
    if att1 not in ATT_NAMES or att2 not in ATT_NAMES:
        return True
    return abs(ATT_NAMES.index(att1) - ATT_NAMES.index(att2)) <= 1    

def possible_same_meaning_predicates(predicate1, predicate2):
    cat1, pos1, col1, pred1_is_complete = decompose_predicate(predicate1)
    cat2, pos2, col2, pred2_is_complete = decompose_predicate(predicate2)

    # Version most strict (need same color/position) and same cat
    is_similar = (cat1 == cat2)
    if (pos1 is not None and pos2 is not None):
        is_similar &= (pos1 == pos2)
    if (col1 is not None and col2 is not None):
        is_similar &= (col1 == col2)
    return is_similar


def possible_close_meaning_predicates(predicate1, predicate2):
    cat1, pos1, col1, pred1_is_complete = decompose_predicate(predicate1)
    cat2, pos2, col2, pred2_is_complete = decompose_predicate(predicate2)

    # Version most strict (need same color/position) and same cat
    is_similar = (cat1 == cat2)
    if (pos1 is not None and pos2 is not None):
        is_similar &= similar_color_or_position(pos1, pos2, POSITION_NAMES)
    if (col1 is not None and col2 is not None):
        is_similar &= similar_color_or_position(col1, col2, COLOR_NAMES)
    return is_similar
  

def all_possible_same_meaning_predicates(list_predicates):
    dict_predicate_sentences_mapping = defaultdict(list)
    one_object_sentences = set()
    for predicate in tqdm.tqdm(list_predicates):
        for sentence, known_predicate in SENTENCE_TO_PREDICATE.items():
            if possible_same_meaning_predicates(predicate, known_predicate):
                dict_predicate_sentences_mapping[predicate].append(sentence)
                one_object_sentences.add(sentence)
    return dict_predicate_sentences_mapping, len(one_object_sentences)


def random_sentence_from_complete_predicate(concept):
    sentences_and_predicates = []
    for sentence, predicate in SENTENCE_TO_PREDICATE.items():
        if possible_same_meaning_predicates(concept, predicate):
            sentences_and_predicates.append([sentence, predicate])
    sentence, predicate = sentences_and_predicates[np.random.choice(len(sentences_and_predicates))]
    return sentence, predicate





# Words
OBJECT_NAMES = []
COLOR_NAMES = []
POSITION_NAMES = []

# Concepts
CATEGORIES = []
POSITIONS = []
COLORS = []
  
# Word to concept mapping
CONCEPT_LISTS = list()
OBJ_NAME_TO_CONCEPT = dict()
COLOR_NAME_TO_CONCEPT = dict()
POSITION_NAME_TO_CONCEPT = dict()

# Vision one-hot encoder
VISION_ENCODER = None

# Sentences mapping related
SENTENCE_TO_PREDICATE = dict()
SENTENCE_TO_ROLES = None

# Max nb of objects seen in an image
LIMIT_NB_OBJECTS_IN_ONE_IMAGE = 2

#create_dataset()
#print(CONCEPT_LISTS)
#print(f'obj_name_dict {OBJ_NAME_TO_CONCEPT}')
#print(f'col_name_dict {COLOR_NAME_TO_CONCEPT}')
