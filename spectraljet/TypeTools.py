import numpy as np
import functools
import awkward as ak

def is_stringy(x):
    if isinstance(x, (str, ak.behaviors.string.StringBehavior)):
        return True
    try:
        comparison = str(x) == x
        if isinstance(comparison, bool):
            return comparison
        else:
            return False
    except ValueError:
        return False


def restring(x):
    if is_stringy(x):
        return ''.join(x)
    return x


class StringyArray(ak.Array):
    """Variation on awkward arrays which respects strings"""
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if is_stringy(item):
            return restring(item)
        try:
            return type(self)(item)
        except TypeError:
            return item

    def __iter__(self):
        for item in super().__iter__():
            if is_stringy(item):
                yield restring(item)
            else:
                try:
                    yield type(self)(item)
                except TypeError:
                    yield item


def is_non_str_iterable(x):
    """
    Check if the given object is any kind of iterable besides a string.

    Parameters
    ----------
    x : object
        Thing to check

    Returns
    -------
    : bool
        is it a non-string iterable?

    """
    return (hasattr(x, '__iter__') and not is_stringy(x))


def soft_equality(a, b):
    """
    Check if all elements of a and b are approximatly equal.
    Either a, b or both may be iterable, apropreate projecttion will be used.

    Parameters
    ----------
    a : object
        to be compared
        
    b : object
        to be compared
        

    Returns
    -------
    : bool or array of bool
        where do a and b match?

    """
    a_iterable = is_non_str_iterable(a)
    b_iterable = is_non_str_iterable(b)
    if not (is_stringy(a) or is_stringy(b)):
        try:
            # old awkward 1.8.0 way
            #aa = np.array(ak.to_list(a)) if a_iterable else a
            #bb = np.array(ak.to_list(b)) if b_iterable else b
            # TODO fix this in awkward
            aa = np.array(list(a)) if a_iterable else a
            bb = np.array(list(b)) if b_iterable else b
            return np.isclose(aa, bb, equal_nan=True)
        except TypeError:
            pass
    if a_iterable and b_iterable:
        if len(a) == len(b):
            return np.fromiter((aa == bb for aa, bb in zip(a, b)),
                               dtype=bool)
        else:
            problem = f"Could not compare length {len(a)} to {len(b)}"
            raise ValueError(problem)
    elif a_iterable:
        b = restring(b)
        return np.fromiter((restring(aa) == b for aa in a),
                           dtype=bool)
    elif b_iterable:
        a = restring(a)
        return np.fromiter((restring(bb) == a for bb in b),
                           dtype=bool)
    else:
        return restring(a) == restring(b)


def generic_sort(iterable):
    generic_key = functools.cmp_to_key(generic_compare)
    return sorted(iterable, key=generic_key)


def generic_compare(a, b):
    a_iterable = is_non_str_iterable(a)
    b_iterable = is_non_str_iterable(b)
    if a_iterable != b_iterable:  # only one is iterable
        return a_iterable
    if a_iterable:  # both are iterable
        for aa, bb in zip(a, b):
            comparison = generic_compare(aa, bb)
            if comparison != 0:
                # the first non matching element breaks the tie
                return comparison
        # there were no non matching elements
        if len(a) == len(b):
            return 0
        return float(len(a) > len(b)) - 0.5
    return generic_non_iterable_compare(a, b)


def generic_non_iterable_compare(a, b):
    string_comp = False
    if is_stringy(a):
        a = restring(a)
        string_comp = True
    if is_stringy(b):
        b = restring(b)
        string_comp = True
    if not string_comp:
        # some things we should try to compare
        # without converting to strings
        try:
            inequality = float(a > b) - 0.5
        except TypeError:
            string_comp = True
    if string_comp:
        inequality = float(str(a) > str(b)) - 0.5
    if a == b:
        return 0
    else:
        return inequality

