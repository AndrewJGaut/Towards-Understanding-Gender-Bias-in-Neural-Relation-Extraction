'''
A class to hold a name and the probability that name occurs
Probabilities will be based off U.S. Census Bureau data
This is for the genderSwap file
'''


class NameProb:

    def __init__(self, name, probability):
        self.name = name
        self.probability = probability

    def getOcc(self):
        return self.name

    def getProb(self):
        return self.probability

    '''
    This makes all NameProb objects sortable
    They are sorted by probability
    '''
    def __lt__(self,other):
        return self.probability < other.probability

    '''
    This makes all NameProb objects hashable
    They are hashed by their occupation
    '''
    def __hash__(self):
        return hash(self.name)

    '''
    This makes NameProb objects comparable
    They are compared by occupation
    Notably, they are NOT compared to another NameProb object but just a string!
    '''
    def __eq__(self, other):
        return self.name == other