# -*- coding: utf-8 -*-





def cat(*dicts):
    """
        Concatenation of two grammars
    """
    
    if len(dicts) == 1:
        return dicts[0]
    
    dict1 = dicts[0]
    dict2 = cat(*dicts[1:])
    
    
    def cat_sentences(s1, s2):
        
        if s1 == '':
            return s2
        if s2 == '':
            return s1
        
        return s1 + ' ' + s2
        
    return {cat_sentences(s1, s2) : r1 + r2 for s1,r1 in dict1.items() for s2,r2 in dict2.items()}
    


def union(*dicts):
    """
        Union of two grammars
    """
    
    items = []
    
    for d in dicts:
        items.extend(d.items())
        
    return dict(items)


def maybe(d):
    """
        In a concatenation, apply maybe to a grammar to make it optional
    """
    return union(d, {'' : []})




def sentence_to_roles(sentence, roles):
    return {sentence : roles}


def role_for_words(role, words):
    return {w : [role] for w in words}


