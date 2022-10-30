from abc import ABC, abstractmethod

class LTM(ABC):
    def __init__(self):
        self.content = {} # dict storing content of the long-term memory

    @abstractmethod
    def p_z_LTM(self, dynamicParams): # function used to sample the probabilistic long-term memory variables
        raise NotImplementedError