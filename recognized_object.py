# -*- coding: utf-8 -*-


################################################################################
#                           Object Returned from vision
################################################################################
 



class RecognizedObject:
    
    def __init__(self, category=None, position=None, color=None):
        
        self.category = category
        self.position = position
        self.color = color
        

    def __str__(self):
        
        pattern = "RecognizedObject('{}', '{}', '{}')"
        
        return pattern.format(self.category, self.position, self.color)
        
    
    def __repr__(self):
        return self.__str__()
    
    
    def to_predicateStr(self):
        return f'{self.position}({self.category},{self.color})'
    
  