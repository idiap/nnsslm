"""
misc.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

from collections.abc import Sequence
from bisect import bisect


def parse_dict(x):
    def _parse_item(y):
        k, v = y.split(':')
        if v.lower() == 'true':
            return k, True
        elif v.lower() == 'false':
            return k, False
        try:
            w = int(v)
        except ValueError:
            try:
                w = float(v)
            except ValueError:
                w = v
        return k, w

    return dict(_parse_item(y) for y in x.split(','))


def pairwise(alist):
    it = iter(alist)
    return list(zip(it, it))


class AllInnerPairs(Sequence):
    """
    All pairs in a set
    """
    def __init__(self, a):
        super().__init__()
        self.a = a

    def _h(self, j):
        return j * (j - 1) // 2

    def _h_inverse(self, index):
        # binary search
        x = 1
        y = len(self.a) + 1
        while y - x > 1:
            m = (x + y) // 2
            if self._h(m) <= index:
                x = m
            else:
                y = m
        return x

    def __len__(self):
        return len(self.a) * (len(self.a) - 1) // 2

    def __getitem__(self, index):
        j = self._h_inverse(index)
        i = index - self._h(j)
        return (self.a[i], self.a[j])


class AllOuterPairs(Sequence):
    """
    All pairs between two sets
    """
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __len__(self):
        return len(self.a) * len(self.b)

    def __getitem__(self, index):
        i = index // len(self.b)
        j = index % len(self.b)
        return (self.a[i], self.b[j])


class InnerPairsDifferentGroups(Sequence):
    """
    All pairs between any different sets
    """
    def __init__(self, l_sets):
        super().__init__()
        self.l_sets = l_sets

        assert len(l_sets) >= 2
        n = len(l_sets[0])  # number of items (previous total)
        self.n_pairs = 0  # number of current pairs
        self.l_n_pairs = [0]
        self.l_n_items = [n]
        for s in l_sets[1:]:
            m = len(s)  # number of items (new)
            self.n_pairs += n * m
            n += m
            self.l_n_items.append(n)
            self.l_n_pairs.append(self.n_pairs)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, index):
        bg = bisect(self.l_n_pairs, index)
        bi = (index - self.l_n_pairs[bg - 1]) // self.l_n_items[bg - 1]
        r = (index - self.l_n_pairs[bg - 1]) % self.l_n_items[bg - 1]
        ag = bisect(self.l_n_items, r)
        ai = r - self.l_n_items[ag]
        return (self.l_sets[ag][ai], self.l_sets[bg][bi])


class OuterPairsDifferentGroup(Sequence):
    """
    any (a, b) in A x B and g(a) != g(b)
    g(a) : group of a
    """
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y), 'not the same number of groups'
        self.x = x
        self.y = y
        self.n_b = sum(len(g) for g in y)

        self.n_pairs = 0
        self.l_n_pairs = []
        offset = 0
        self.b_offset = []
        for i in range(len(x)):
            self.n_pairs += len(x[i]) * (self.n_b - len(y[i]))
            self.l_n_pairs.append(self.n_pairs)
            offset += len(y[i])
            self.b_offset.append(offset)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, index):
        ag = bisect(self.l_n_pairs, index)
        r = index - self.l_n_pairs[ag]
        ai = r // (self.n_b - len(self.y[ag]))
        r = ((r % (self.n_b - len(self.y[ag]))) + self.b_offset[ag]) % self.n_b
        bg = bisect(self.b_offset, r)
        bi = r - self.b_offset[bg]

        return (self.x[ag][ai], self.y[bg][bi])


def sample_no_replace(a, n, rng):
    if n < len(a):
        return rng.sample(a, n)
    else:
        return a


def sample_pairs(a, n_pairs, rng):
    if n_pairs * 2 <= len(a):
        l_pairs = pairwise(rng.sample(a, n_pairs * 2))
        assert len(l_pairs) == n_pairs
    else:
        candiates = AllInnerPairs(a)
        l_pairs = sample_no_replace(candiates, n_pairs, rng)
        assert len(l_pairs) == min(n_pairs, len(candiates))
    return l_pairs


def sample_outer_pairs(a, b, n_pairs, rng):
    candiates = AllOuterPairs(a, b)
    l_pairs = sample_no_replace(candiates, n_pairs, rng)
    assert len(l_pairs) == min(n_pairs, len(candiates))
    return l_pairs


def sample_pairs_from_groups(l_a, n_pairs, rng):
    candiates = InnerPairsDifferentGroups(l_a)
    l_pairs = sample_no_replace(candiates, n_pairs, rng)
    assert len(l_pairs) == min(n_pairs, len(candiates))
    return l_pairs


def sample_outer_pairs_from_groups(l_a, l_b, n_pairs, rng):
    candiates = OuterPairsDifferentGroup(l_a, l_b)
    l_pairs = sample_no_replace(candiates, n_pairs, rng)
    assert len(l_pairs) == min(n_pairs, len(candiates))
    return l_pairs
