# Provided the package is run as a module, this shouldn't be needed.
# try calling with `pytest` from the directory above
# import sys
# from pathlib import Path
# path_root1 = Path(__file__).parents[1]
# sys.path.append(str(path_root1))

from ..spectraljet import TypeTools
from numpy import testing as tst
import numpy as np

def test_is_non_str_iterable():
    assert TypeTools.is_non_str_iterable([])
    assert TypeTools.is_non_str_iterable([2])
    assert TypeTools.is_non_str_iterable(np.ones(3))
    assert TypeTools.is_non_str_iterable({4, 5})
    assert not TypeTools.is_non_str_iterable("{4, 5}")
    assert not TypeTools.is_non_str_iterable(4)


def test_soft_equality():
    assert TypeTools.soft_equality(1, 1.)
    assert TypeTools.soft_equality("dog", "dog")
    assert not TypeTools.soft_equality("apples", "oranges")
    assert not TypeTools.soft_equality(1., 2.)
    assert np.all(TypeTools.soft_equality([1, 2], [1, 2]))
    assert np.all(TypeTools.soft_equality(np.array([1., 2.]), [1, 2]))
    tst.assert_allclose(TypeTools.soft_equality([1, 2], 1),
                        [True, False])
    tst.assert_allclose(TypeTools.soft_equality(1, [1, 2]),
                        [True, False])

