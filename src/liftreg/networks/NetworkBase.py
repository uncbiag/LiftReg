"""Base class for Network classes. 

This class behaves as an interface for Network classes.
Every inheriented class needs to implemente initialize,
_train_model and _test_model methods.

"""
from abc import ABC, abstractmethod
import torch

class NetworkBase(ABC):
    PHASES = ['train', 'val', 'debug']

    def __init__(self):
        self.model = None
        self.is_train = True

    def run(self):
        if self.mode == 'train':
            self._train_model()
        elif self.mode == 'test':
            self._test_model()

    @abstractmethod
    def initialize(self, setting):
        NotImplemented
    
    @abstractmethod
    def _train_model(self):
        NotImplemented

    @abstractmethod
    def _test_model(self):
        NotImplemented
    
    def set_train(self):
        """
        set the model in train mode (only for learning methods)
        :return:
        """
        self.model.train()
        self.is_train = True
    
    def set_val(self):
        """
        set the model in validation mode (only for learning methods)
        :return:
        """
        self.model.eval()
        self.is_train = False

    def set_debug(self):
        """
        set the model in debug (subset of training set) mode (only for learning methods)
        :return:
        """
        self.model.eval()
        self.is_train = False

    def set_test(self):
        """
        set the model in test mode ( only for learning methods)
        :return:
        """
        self.model.eval()
        self.is_train = False

    