#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:45:16 2019

@author: ajuven
"""

NO_ROLE = 0
ACTION = 1
OBJECT = 2
COLOR = 3
NB_ROLES = 4



class WordPredicate:
    


    def __init__(self, sentence, role_list):

        if sentence is None and role_list is None:
            self.action = None
            self.object = None
            self.color = None
            return
        
        words = sentence.split(' ')
        found_roles = {x : None for x in [ACTION, OBJECT, COLOR]}
        
        self.is_invalid = False
        
        for i, r  in enumerate(role_list):
            
            if r == NO_ROLE:
                continue
            
            if found_roles[r] is not None:
                self.is_invalid = True
                return
            
            found_roles[r] = words[i]
            
        if found_roles[ACTION] is None or found_roles[OBJECT] is None:
            self.is_invalid = True
            return
            
        self.action = found_roles[ACTION]
        self.object = found_roles[OBJECT]
        self.color = found_roles[COLOR]


    @classmethod
    def from_string(cls, predicate_str):
        cat, pos, col = decompose_predicate_components(predicate_str)
        return WordPredicate.from_list_attributes(attributes=[cat, pos, col])


    @classmethod
    def from_list_attributes(cls, attributes: list):

        if len(attributes) != 3:
            raise TypeError("Attributes do not contain the right parameters")
        
        obj, position, color = attributes
        pseudo_sentence = f'{obj} {position} {color}'
        role_list = [OBJECT, ACTION, COLOR]
        
        word_predicate = cls(pseudo_sentence, role_list)
        
        return word_predicate 


    def __str__(self):

        # if self.object is None and self.action is None and self.color is None:
        #     return 'None'

        # if self.is_invalid:
        #     return 'INVALID'        
        
        # if self.color is None:
        #     return f'{self.action}({self.object})'
        
        return f'{self.action}({self.object},{self.color})'


    def __repr__(self):
        return self.__str__()
    
    
    def self_expose(self):
        return f'<{self.object}_obj>', f'<{self.action}_pos>', f'<{self.color}_col>'
    
    
    def __eq__(self, predicate):
        return predicate is not None \
               and self.object == predicate.object \
               and self.action == predicate.action \
               and self.color == predicate.color
        

    def pos_le(self, predicate):
        if predicate is None:
            return False
        return self.quantize_position() <= predicate.quantize_position()
               
               
    def quantize_position(self):
        if self.action is None or self.action == 'middle':
            return 1
        elif self.action == 'left':
            return 0
        elif self.action == 'right':
            return 2




def possible_combinations_of_complete_predicates(position_names, object_names, color_names):
    possible_predicates = set()
    for pos in position_names:
        for obj in object_names:
            for col in color_names:
                possible_predicates.add(f'{pos}({obj},{col})')
    return possible_predicates


def decompose_predicate_components(predicate):
    
    # Transform predicate to string and strip all spaces
    pred = str(predicate).replace(' ','')
    if pred.lower() == 'none':
        return None, None, None
    
    # Split predicate into list of elementary components
    components = pred[:-1].replace('(',',').split(',')
    
    # Parse the list of components into category, color and position
    position = components[0]
    category = components[1]
    if len(components) == 3:
        color = components[2] if components[2].lower() != 'none' else None
    else:
        color = None

    return category, position, color


def filtered_decompose_predicate(predicate, position_names):
    
    # Decompose the predicate directly to elementary components
    category, position, color = decompose_predicate_components(predicate)
    
    # Filter unknown component's attributes
    is_complete_predicate = True
    if position not in position_names:
        position = None
    if position is None or color is None:
        is_complete_predicate = False
    
    return category, position, color, is_complete_predicate


def is_complete_predicate(predicate, position_names):
    _, _, _, is_complete = filtered_decompose_predicate(predicate, position_names)
    return is_complete 


def exact_same_predicates(predicate1, predicate2, *filtered_list):
    position_names = filtered_list[0]
    cat1, pos1, col1, _ = filtered_decompose_predicate(predicate1, position_names)
    cat2, pos2, col2, _ = filtered_decompose_predicate(predicate2, position_names)
    return cat1 == cat2 and pos1 == pos2 and col1 == col2    


def possible_same_meaning_predicates(predicate1, predicate2, *filtered_list):
    position_names = filtered_list[0]
    cat1, pos1, col1, pred1_is_complete = filtered_decompose_predicate(predicate1, position_names)
    cat2, pos2, col2, pred2_is_complete = filtered_decompose_predicate(predicate2, position_names)

    # Version most strict (need same color/position) and same cat
    is_similar = (cat1 == cat2)
    if (pos1 is not None and pos2 is not None):
        is_similar &= (pos1 == pos2)
    if (col1 is not None and col2 is not None):
        is_similar &= (col1 == col2)
    return is_similar


def possible_close_meaning_predicates(predicate1, predicate2, *filtered_list):
    position_names, color_names = filtered_list[:2]
    cat1, pos1, col1, pred1_is_complete = filtered_decompose_predicate(predicate1, position_names)
    cat2, pos2, col2, pred2_is_complete = filtered_decompose_predicate(predicate2, position_names)

    # Version most strict (need same color/position) and same cat
    is_similar = (cat1 == cat2)
    if (pos1 is not None and pos2 is not None):
        is_similar &= similar_color_or_position(pos1, pos2, position_names)
    if (col1 is not None and col2 is not None):
        is_similar &= similar_color_or_position(col1, col2, color_names)
    return is_similar


def similar_color_or_position(att1, att2, ATT_NAMES):
    if att1 not in ATT_NAMES or att2 not in ATT_NAMES:
        return True
    return abs(ATT_NAMES.index(att1) - ATT_NAMES.index(att2)) <= 1
