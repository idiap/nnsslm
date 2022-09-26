"""
utils.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""


class HasNextIter(object):
    def __init__(self, iterable):
        self.it = iter(iterable)
        self.cache = []

    def __next__(self):
        if self.cache:
            return self.cache.pop()
        else:
            return next(self.it)

    def has_next(self):
        if self.cache:
            return True
        else:
            try:
                self.cache.append(next(self.it))
                return True
            except StopIteration:
                return False
