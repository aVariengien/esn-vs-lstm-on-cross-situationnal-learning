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
            self.is_invalid = True
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


    def __str__(self):
        
        if self.is_invalid:
            return 'INVALID'
        
        if self.color is None:
            return self.action + '(' + self.object + ')'
        
        return self.action + '(' + self.object + ', ' + self.color + ')'


    def __repr__(self):
        return self.__str__()
        



