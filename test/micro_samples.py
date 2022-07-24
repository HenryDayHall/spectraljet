import awkward as ak
import numpy as np


class AwkdArrays:
    empty = ak.from_iter([])
    one_one = ak.from_iter([1])
    minus_plus = ak.from_iter(np.arange(-2, 3))
    event_ints = ak.from_iter([[1, 2], [3]])
    jet_ints = ak.from_iter([ak.from_iter([[1, 2], [3]]),
                             ak.from_iter([[4,5,6]])])
    event_floats = ak.from_iter([[.1, .2], [.3]])
    jet_floats = ak.from_iter([ak.from_iter([[.1, .2], [.3]]),
                               ak.from_iter([[.4,.5,.6]])])
    empty_event = ak.from_iter([[], [1]])
    empty_jet = ak.from_iter([ak.from_iter([[], [1]]),
                              ak.from_iter([[2, 3]])])
    empty_events = ak.from_iter([[], []])
    empty_jets = ak.from_iter([ak.from_iter([[],[]]),
                               ak.from_iter([[]])])


class SimpleClusterSamples:
    config_1 = {'DeltaR': 1., 'ExpofPTMultiplier': 0}
    config_2 = {'DeltaR': 1., 'ExpofPTMultiplier': 1}
    config_3 = {'DeltaR': 1., 'ExpofPTMultiplier': -1}
    config_4 = {'DeltaR': .4, 'ExpofPTMultiplier': 0}
    config_5 = {'DeltaR': .4, 'ExpofPTMultiplier': 1}
    config_6 = {'DeltaR': .4, 'ExpofPTMultiplier': -1}
    # there is no garantee on left right child order, or global order of pseudojets
    unitless = True   # do we work with a unitless version of distance
    empty_inp = {'ints': np.array([]).reshape((-1, 5)), 'floats': np.array([]).reshape((-1, 8))}
    one_inp = {'ints': np.array([[0, -1, -1, -1, -1]]),
               'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 0.]])}
    two_degenerate = {'ints': np.array([[0, -1, -1, -1, -1],
                                        [1, -1, -1, -1, -1]]),
                      'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 0.],
                                          [1., 0., 0., 1., 1., 0., 0., 0., 1.]])}
    degenerate_join = {'ints': np.array([[0, 2, -1, -1, -1],
                                         [1, 2, -1, -1, -1],
                                         [2, -1, 0, 1, 0]]),
                       'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 0.],
                                           [1., 0., 0., 1., 1., 0., 0., 0., 1.],
                                           [2., 0., 0., 2., 2., 0., 0., 0., 1.]])}
    two_close = {'ints': np.array([[0, -1, -1, -1, -1],
                                   [1, -1, -1, -1, -1]]),
                 'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 1.],
                                     [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0., 1.]])}
    close_join = {'ints': np.array([[0, 2, -1, -1, -1],
                                    [1, 2, -1, -1, -1],
                                    [2, -1, 0, 1, 0]]),
                  'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 1.],
                                      [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0., 1.],
                                      [2.*np.cos(0.05), 0., 0.05, 2., 1. + np.cos(0.1), np.sin(0.1), 0., 0.1, 2.]])}
    two_oposite = {'ints': np.array([[0, -1, -1, -1, -1],
                                     [1, -1, -1, -1, -1]]),
                   'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0., 0.],
                                       [1., 0., np.pi, 1., -1., 0., 0., 0., 0.]])}

