""" Module to test the new FormJets module """
import sys
from pathlib import Path
path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))
import scipy.spatial
import pytest
import warnings
import os
import awkward as ak
from .tools import TempTestDir
from .micro_samples import SimpleClusterSamples
import numpy as np
from numpy import testing as tst
from spectraljet import FormJets
from spectraljet import Components
from spectraljet import FastJetPython


def match_ints(ints1, ints2):
    """ find out if one array of ints in infact a reshuffle of the other
    get shuffle order. Is independent of the order of child 1 and 2"""
    ints1 = np.array(ints1)
    ints2 = np.array(ints2)
    if ints1.shape != ints2.shape:
        assert False, f"Ints dont have the same shape; {ints1.shape}, {ints2.shape}"
    child_columns = slice(2, 4)
    ints1.T[child_columns] = np.sort(ints1.T[child_columns], axis=0)
    ints2.T[child_columns] = np.sort(ints2.T[child_columns], axis=0)
    avalible = list(range(ints1.shape[0]))  # make sure we match one to one
    order = []
    for row in ints2:
        matches = np.where(np.all(ints1[avalible] == row, axis=1))[0]
        if len(matches) == 0:
            assert False, f"Ints don't match. \n{ints1}\n~~~\n{ints2}"
        picked = avalible.pop(matches[0])
        order.append(picked)
    return order


def match_ints_floats(ints1, floats1, ints2, floats2, compare_distance=True,
                      compare_size=True, distance_modifier=1.):
    ints_order = match_ints(ints1, ints2)
    floats1 = np.array(floats1)[ints_order]
    floats2 = np.array(floats2)
    # make sure the phi coordinates are -np.pi to pi
    floats1[:, 2] = Components.confine_angle(floats1[:, 2])
    floats2[:, 2] = Components.confine_angle(floats2[:, 2])
    if compare_size:
        tst.assert_allclose(floats1[:, -1], floats2[:, -1],
                            atol=0.0005, err_msg="Sizes don't match")
    if compare_distance:
        floats2[:, -2] *= distance_modifier
        tst.assert_allclose(floats1[:, -2], floats2[:, -2],
                            atol=0.0005, err_msg="Distances don't match")
    floats1 = floats1[:, :-2]
    floats2 = floats2[:, :-2]
    tst.assert_allclose(floats1, floats2, atol=0.0005, err_msg="Floats don't match")


def set_JetInputs(eventWise, floats):
    columns = ["JetInputs_" + name for name in FormJets.Agglomerative.float_columns
               if "Distance" not in name]
    if len(floats):
        contents = {name: ak.from_iter([floats[:, i]]) for i, name in enumerate(columns)}
    else:
        contents = {name: ak.from_iter([[]]) for i, name in enumerate(columns)}
    columns.append("JetInputs_SourceIdx")
    contents["JetInputs_SourceIdx"] = ak.from_iter([np.arange(len(floats))])
    eventWise.append(**contents)


# setup and test indervidual methods inside the jet classes
def make_simple_jets(floats, jet_params={}, jet_class=FormJets.GeneralisedKT,
                     run=False, **kwargs):
    with TempTestDir("tst") as dir_name:
        ew = Components.EventWise(os.path.join(dir_name, "tmp.parquet"))
        set_JetInputs(ew, floats)
        ew.selected_event = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        jets = jet_class(ew, dict_jet_params=jet_params, run=run, **kwargs)
    return jets


def test_knn():
    # zero points should not create an error
    points = []
    expected = np.array([]).reshape((0, 0))
    distances = np.array([]).reshape((0, 0))
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # one point is always it's own nearest neigbour
    points = [1]
    expected = np.array([[True]])
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # even if no neighbours are requested
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    # two points
    points = [1, 2]
    expected = np.full((2,2), True)
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # without neighbours
    expected = np.full((2,2), False)
    np.fill_diagonal(expected, True)
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    # three points
    points = [3, 1, 2.5]
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    expected = np.full((3,3), False)
    np.fill_diagonal(expected, True)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    expected = np.array([[True, False, True],
                         [False, True, True],
                         [True, True, True]])
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    expected = np.full((3,3), True)
    tst.assert_allclose(FormJets.knn(distances, 2),  expected)


def test_return_one():
    """Used to make a no-op"""
    out = FormJets.return_one(5)
    assert out == 1.


def test_ca_distances2():
    """ Distances in physical space according to the Cambridge Aachen metric.

    Parameters
    ----------
    rapidity : array of float
        Row of rapidity values.
    phi : array of float
        Row of phi values.
    rapidity_column : 2d array of float (optional)
        Column of rapidity values.
        If not given, taken as the transpose of the row.
    phi_column : 2d array of float (optional)
        Column of phi values.
        If not given, taken as the transpose of the row.
        
    Returns
    -------
    distances2 : 2d array of float
        Distances squared between points in the row and the column.
    """
    # shouldn't choke on empty input
    out = FormJets.ca_distances2(np.ones(0), np.ones(0))
    assert len(out) == 0
    # or input lenght 1
    out = FormJets.ca_distances2(np.ones(1), np.ones(1))
    assert len(out) == 1
    # should give good ansers otherwise
    out = FormJets.ca_distances2(np.ones(2), np.ones(2))
    tst.assert_allclose(out, np.zeros((2, 2)))
    pis = np.array([np.pi, -np.pi])
    out = FormJets.ca_distances2(np.ones(2), pis)
    tst.assert_allclose(out, np.zeros((2, 2)))
    out = FormJets.ca_distances2(np.array([0, 1]), pis)
    tst.assert_allclose(out, np.array([[0., 1.], [1., 0.]]))
    # also should be able to use columns
    out = FormJets.ca_distances2(np.ones(2), np.ones(2),
                                 np.ones((1, 1)), np.ones((1, 1)))
    tst.assert_allclose(out, np.zeros((1, 2)))


def test_genkt_factor():
    """A gen-kt factor, which will be used to reduce
    affinity to particles with low pt.

    Parameters
    ----------
    exponant : float
        power to raise each pt to.
    pt : array of float
        Row of pt values.
    pt_column : 2d array of float (optional)
        Column of pt values.
        If not given, taken as the transpose of the row.
        
    Returns
    -------
    factor : 2d array of float
        PT factors between points in the row and the column.
    """
    # shouldn't choke on empty input
    out = FormJets.genkt_factor(0., np.ones(0))
    assert len(out) == 0
    # or input lenght 1
    out = FormJets.genkt_factor(5., np.ones(1))
    tst.assert_allclose(out, np.ones((1, 1)))
    # should give good ansers otherwise
    out = FormJets.genkt_factor(2., np.array([1, 2]))
    tst.assert_allclose(out, np.array([[1., 1.], [1., 4.]]))
    # also should be able to use columns
    out = FormJets.genkt_factor(2., np.array([1, 3]), np.array([[2]]))
    tst.assert_allclose(out, np.array([[1., 4.]]))


def test_ratiokt_factor():
    """A ratio factor, which will be used to reduce affinity
    to particles with differing pt.

    Parameters
    ----------
    exponant : float
        power to raise each pt to.
    pt : array of float
        Row of pt values.
    pt_column : 2d array of float (optional)
        Column of pt values.
        If not given, taken as the transpose of the row.
        
    Returns
    -------
    factor : 2d array of float
        PT factors between points in the row and the column.
    """
    # shouldn't choke on empty input
    out = FormJets.ratiokt_factor(0., np.ones(0))
    assert len(out) == 0
    # or input lenght 1
    out = FormJets.ratiokt_factor(5., np.ones(1))
    tst.assert_allclose(out, np.ones((1, 1)))
    # should give good ansers otherwise
    out = FormJets.ratiokt_factor(2., np.array([1, 2]))
    tst.assert_allclose(out, np.array([[1., 0.25], [0.25, 1.]]))
    # also should be able to use columns
    out = FormJets.ratiokt_factor(2., np.array([1, 3]), np.array([[2]]))
    tst.assert_allclose(out, np.array([[0.25, 4./9.]]))


class TestExpAffinity:
    def function(self, *args, **kwargs):
        """Calculating the affinity from a_ij = exp(-d_ij^exponent/sigma)

        Parameters
        -------
        distances2 : 2d array of float
            Distances squared between points.
        sigma : float (optional)
            Controls the bandwidth of the kernal.
            (Default; 1)
        exponant : float (optional)
            power to raise the distance to.
            (Default; 2)
        fill_diagonal : bool
            If true, fill the diagonal with 0.
            
        Returns
        -------
        aff : 2d array of float
            Affinities between the points provided.
        """
        return FormJets.exp_affinity(*args, **kwargs)

    def test_exp_affinity(self):
        # shouldn't choke on empty input
        out = self.function(np.zeros((0, 0)))
        assert len(out) == 0
        # or input lenght 1
        out = self.function(np.zeros((1, 1)))
        tst.assert_allclose(out, np.zeros((1, 1)))
        # should give good ansers otherwise
        out = self.function(np.array([[1, 2], [2, 1]])**2, 5.)
        tst.assert_allclose(out, np.array([[0., np.exp(-4/5.)],
                                           [np.exp(-4/5.), 0.]]))

    def test_assymetric(self):
        out = self.function(np.array([[1, 2], [3, 0]])**2, 5.)
        tst.assert_allclose(out, np.array([[0., np.exp(-4/5.)],
                                           [np.exp(-9/5.), 0.]]))

    def test_exponent_fill_diagonal(self):
        out = self.function(np.array([[1, 2], [3, 0]]), 4., 3.,
                                    fill_diagonal=False)
        tst.assert_allclose(out, np.array([[np.exp(-1/4), np.exp(-(2**1.5)/4)],
                                           [np.exp(-(3**1.5)/4), 1.]]))


def test_unnormed_laplacian():
    """Construct an unnormed laplacian, L = D-A

    Parameters
    -------
    affinities : 2d array of float
        Square grid of affinities between points.
        
    Returns
    -------
    laplacian : 2d array of float
        Laplacian of the graph.
    """
    # shouldn't choke on empty input
    out = FormJets.unnormed_laplacian(np.zeros((0, 0)))
    assert len(out) == 0
    # or input lenght 1
    out = FormJets.unnormed_laplacian(np.ones((1, 1)))
    tst.assert_allclose(out, np.zeros((1, 1)))
    # should give good ansers otherwise
    out = FormJets.unnormed_laplacian(np.array([[1, 2], [2, 1]]))
    tst.assert_allclose(out, np.array([[2, -2], [-2, 2]]))


def test_normalised_laplacian():
    """Construct a laplacian with arbitary norm,
    Z=norm_by, L = Z^-1/2(D-A)Z^-1/2

    Parameters
    -------
    affinities : 2d array of float
        Square grid of affinities between points.
    norm_by : 2d or 1d array of floats
        
    Returns
    -------
    laplacian : 2d array of float
        Laplacian of the graph.
    """
    # shouldn't choke on empty input
    out = FormJets.normalised_laplacian(np.zeros((0, 0)), np.ones(0))
    assert len(out) == 0
    # or input lenght 1
    out = FormJets.normalised_laplacian(np.ones((1, 1)), np.ones(1))
    tst.assert_allclose(out, np.zeros((1, 1)))
    # should give good ansers otherwise
    out = FormJets.normalised_laplacian(np.array([[1, 2], [2, 1]]),
                                        np.array([1, 2]))
    tst.assert_allclose(out, np.array([[2, -2/np.sqrt(2)],
                                       [-2/np.sqrt(2), 1]]))


class TestSymmetricLaplacian:
    def function(self, affinities):
        """Construct a symmetric laplacian, L = D^-1/2(D-A)D^-1/2

        Parameters
        -------
        affinities : 2d array of float
            Square grid of affinities between points.
            
        Returns
        -------
        laplacian : 2d array of float
            Laplacian of the graph.
        """
        return FormJets.symmetric_laplacian(affinities)

    def test_symmetric_laplacian(self):
        # shouldn't choke on empty input
        out = self.function(np.zeros((0, 0)))
        assert len(out) == 0
        # or input lenght 1
        out = self.function(np.ones((1, 1)))
        tst.assert_allclose(out, np.zeros((1, 1)))
        # should give good ansers otherwise
        out = self.function(np.array([[1, 2], [2, 1]]))
        tst.assert_allclose(out, np.array([[2/3, -2/3],
                                           [-2/3, 2/3]]))


def test_embedding_space():
    """Calculate eigenvectors of the laplacian to make the embedding space

    Parameters
    -------
    laplacian : 2d array of float
        Laplacian of the graph.
    num_dims : int (optional)
        Max number of dimensions in the embedding space.
        If None, no max dimensions.
        (Default; None)
    max_eigval: float (optional)
        Max value of eigenvalue assocated with dimensions' eigenvector
        in the embedding space.  If None, no max eigenvalue.
        (Default; None)
        
    Returns
    -------
    values : array of float
        Eigenvalue of the dimensions
    vectors : 2d array of float
        Eigenvectors of the dimensions
    """
    laplacian = np.arange(3*3).reshape((3, 3))
    laplacian += laplacian.T
    laplacian = np.diag(np.sum(laplacian, 0)) - laplacian
    vals, vectors = FormJets.embedding_space(laplacian, 2)
    assert len(vals) == 2
    rhs = np.sum(laplacian * vectors[:, 0], axis=1)
    lhs = vals[0] * vectors[:, 0]
    tst.assert_allclose(rhs, lhs)
    vals, vectors = FormJets.embedding_space(laplacian, max_eigval=0.1)
    assert np.all(vals < 0.1)
    # if the laplacien is two disconnected subgraphs,
    # then there shoudl be two 0 eigenvalues
    laplacian = np.zeros((4, 4))
    np.fill_diagonal(laplacian, 1)
    laplacian[[0, 1, 2, 3], [1, 0, 3, 2]] = -1
    vals, vectors = FormJets.embedding_space(laplacian, max_eigval=0.1)
    tst.assert_allclose(vals[0], 0., atol=1e-4)
    tst.assert_allclose(vectors[[0, 2], 0], vectors[[1, 3], 0])
    vals, vectors = FormJets.embedding_space(laplacian, num_dims=1)
    tst.assert_allclose(vals[0], 0., atol=1e-4)
    tst.assert_allclose(vectors[[0, 2], 0], vectors[[1, 3], 0])

    
def test_dimension_hardness():
    """Calculate the hardness factors of each dimensions of the embedding space.
    Used to ensure dimensions with soft particles are small.

    Parameters
    ----------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space
    pt : (n, ) array of floats
        pt of the particles

    Returns
    -------
    hardness : (m, ) array of floats
        the hardness of each dimension
    """
    embedding = np.array([[2., 4.],
                          [0., 0.],
                          [1., -1.]])
    pt = np.array([1., 10., 2.])
    out = FormJets.dimension_hardness(embedding, pt)
    tst.assert_allclose(out, np.array([[4./3., 6./3.]]))


def test_embedding_norm():
    """Normalise all points in the embedding to length 1.
    Paper that also does this version of distance in embedding space
    https://projecteuclid.org/journals/statistical-science/volume-23/issue-3/Multiway-Spectral-Clustering-A-Margin-Based-Perspective/10.1214/08-STS266.full

    Parameters
    ----------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space

    Returns
    -------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space
    """
    embedding = np.array([[2., 4.],
                          [-1., 0.],
                          [1., -1.]])
    out = FormJets.embedding_norm(embedding)
    tst.assert_allclose(out, np.array([[1/np.sqrt(5), 2/np.sqrt(5)],
                                       [-1.,  0.],
                                       [np.sqrt(0.5), -np.sqrt(0.5)]]))


def test_embedding_distance2():
    """Distances squared in the embedding space.

    Parameters
    ----------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space

    Returns
    -------
    distances2 : (n, n) array of floats
        distances between particles squared
    """
    embedding = np.array([[2., 4.],
                          [-1., 0.],
                          [1., -1.]])
    out = FormJets.embedding_distance2(embedding)
    tst.assert_allclose(out, 
                        np.array([[np.inf, 9+16, 25+1],
                                  [9+16, np.inf, 4+1],
                                  [25+1, 4+1, np.inf]]))


def test_embedding_angular2():
    embedding = np.array([[0., 1.],
                          [0., 2.],
                          [1., 0.],
                          [0., -1]])
    out = FormJets.embedding_angular2(embedding)
    tst.assert_allclose(out, 
                        np.array([[np.inf, 0., np.pi/2, np.pi],
                                  [0., np.inf, np.pi/2, np.pi],
                                  [np.pi/2, np.pi/2, np.inf, np.pi/2],
                                  [np.pi, np.pi, np.pi/2, np.inf]])**2)


# class Agglomerative

def test_Agglomerative_setup_ints_floats():
    """ Create the _ints and _floats, along with 
    the _avaliable_mask and _avaliable_idxs

    Parameters
    ----------
    input_data : EventWise or (2d array of ints, 2d array of floats)
        data file for inputs
    """
    floats = SimpleClusterSamples.two_close["floats"]
    with TempTestDir("tst") as dir_name:
        ew = Components.EventWise(os.path.join(dir_name, "tmp.parquet"))
        ew.selected_event = 0
        set_JetInputs(ew, floats)
        agg = FormJets.GeneralisedKT(ew)
        # should make 3 rows so there is a row for the 
        tst.assert_allclose(agg._avaliable_mask, [True, True, False])
        tst.assert_allclose(agg.PT[:2], np.ones(2))
        assert {0, 1} == set(agg._avaliable_idxs)
        assert set(agg.Label) == {-1, 0, 1}
        # should be able to achive the same by passing ints and floats
        ints = -np.ones_like(agg._ints)
        ints[[0, 1], 0] = [0, 1]
        agg2 = FormJets.GeneralisedKT((agg._ints[:2], floats))
        tst.assert_allclose(agg2._ints, agg._ints)
        tst.assert_allclose(agg2._floats, agg._floats)


def test_Agglomerative_2d_avaliable_indices():
    """
    Using the _avaliable_idxs make indices for indexing
    the corrisponding minor or a 2d matrix.

    Returns
    -------
    : tuple of arrays
        tuple that will index the matrix minor
    """
    ints = -np.ones((4, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2, 3]
    floats = np.zeros((4, len(FormJets.Agglomerative.float_columns)))
    agg = FormJets.GeneralisedKT((ints, floats))
    agg._update_avalible([1, 3])
    mask = agg._2d_avaliable_indices
    test = np.array([[0, 1, 2, 3],
                     [4, 5, 6, 7],
                     [8, 9, 10, 11],
                     [12, 13, 14, 15]])
    tst.assert_allclose(test[mask], np.array([[0, 2], [8, 10]]))


def test_Agglomerative_reoptimise_preallocated():
    """Rearange the objects in memory to accomidate more.

    Memory limit has been reached, the preallocated arrays
    need to be rearanged to allow for removing objects which
    are no longer needed.
    anything still in _avaliable_idxs will not be moved.
    Also, remove anything in debug_data, becuase it will be
    invalidated.
    """
    floats = SimpleClusterSamples.two_close["floats"]
    floats = np.concatenate((floats, floats))
    floats[:, FormJets.Clustering.float_columns.index("Energy")] = \
        np.arange(4) + 1
    ints = -np.ones((4, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2, 3]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=5)
    agg.run()
    assert len(agg.Label) == 7
    assert np.sum(agg.Label > -1) == 7
    assert len(agg.Available_Label) == 1
    label = list(agg.Label)
    energies = [agg.Energy[label.index(i)] for i in range(4)]
    tst.assert_allclose(energies, [1., 2., 3., 4.])


def test_Agglomerative_get_historic_2d_mask():
    """get a _2d_avaliable_indices mask for a previous step

    only works if debug_data is stored

    Parameters
    ----------
    step : int
        index of step starting from oldest retained step
        which may not be first step

    Returns
    -------
    : tuple of arrays
        tuple that will index the matrix minor
    """
    floats = SimpleClusterSamples.two_close["floats"]
    floats = np.concatenate((floats, floats))
    ints = -np.ones((4, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2, 3]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    agg.debug_run()
    first_mask = agg.get_historic_2d_mask(0)
    test = np.arange(100).reshape((10, 10))
    selected = test[first_mask]
    expected = np.array([[0, 1, 2, 3],
                         [10, 11, 12, 13],
                         [20, 21, 22, 23],
                         [30, 31, 32, 33]])
    tst.assert_allclose(expected, selected)


def test_Agglomerative_update_avalible():
    """Update which indices are avalible

    Parameters
    ----------
    idxs_out : iterable of ints
        the indices of points that are no longer avaliable.
    idxs_in : iterable of ints (optional)
        the indices of points that are now avaliable.
    """
    floats = SimpleClusterSamples.two_close["floats"]
    floats = np.concatenate((floats, floats))
    ints = -np.ones((4, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2, 3]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    agg._update_avalible([0])
    assert set(agg._avaliable_idxs) == {1, 2, 3}
    expected = np.zeros(10, dtype=bool)
    expected[[1, 2, 3]] = True
    tst.assert_allclose(expected, agg._avaliable_mask)
    agg._update_avalible([], [5, 6])
    assert set(agg._avaliable_idxs) == {1, 2, 3, 5, 6}
    expected[[5, 6]] = True
    tst.assert_allclose(expected, agg._avaliable_mask)


def test_Agglomerative_next_free_row():
    """Find the next free index to place a new point.

    Returns
    -------
    i : int
        index of free point
    """
    floats = SimpleClusterSamples.two_close["floats"]
    floats = np.concatenate((floats, floats))
    ints = -np.ones((4, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2, 3]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    assert agg._next_free_row() == 4
    agg._ints[0, 0] = -1
    assert agg._next_free_row() == 0
    agg._ints[:, 0] = np.arange(10)
    assert agg._next_free_row() == 10


def test_Agglomerative_chose_pair():
    """ Find the next two particles to join.

    Return
    ----------
    row : int
        index of first of the pair of particles to next join.
    column : int
        index of second of the pair of particles to next join.
    """
    floats = SimpleClusterSamples.two_close["floats"]
    floats = np.concatenate((floats, floats[[0]]))
    ints = -np.ones((3, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1, 2]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    agg.setup_internal()
    idx1, idx2 = agg.chose_pair()
    assert {0, 2} == {idx1, idx2}

    floats = SimpleClusterSamples.two_oposite["floats"]
    params = {"DeltaR": 100}
    ints = -np.ones((2, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = [0, 1]
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10,
                                 dict_jet_params=params)
    agg.setup_internal()
    idx1, idx2 = agg.chose_pair()
    assert {0, 1} == {idx1, idx2}


def test_Agglomerative_combine_ints_floats():
    """
    Caluclate the floats and ints created by combining two pseudojets.

    Parameters
    ----------
    idx1 : int
        index of the first pseudojet to input
    idx2 : int
        index of the second pseudojet to input
    distance2 : float
        distanc esquared between the pseudojets

    Returns
    -------
    ints : list of ints
        int columns of the combined pseudojet,
        order as per the column attributes
    floats : list of floats
        float columns of the combined pseudojet,
        order as per the column attributes
    """
    degenerate_ints = SimpleClusterSamples.degenerate_join["ints"]
    degenerate_floats = SimpleClusterSamples.degenerate_join["floats"]
    agg = FormJets.GeneralisedKT((degenerate_ints[:2], degenerate_floats[:2]),
                                 memory_cap=10)
    new_ints, new_floats = agg.combine_ints_floats(0, 1, 0.)
    tst.assert_allclose(new_ints, degenerate_ints[-1])
    tst.assert_allclose(new_floats, degenerate_floats[-1])

    close_ints = SimpleClusterSamples.close_join["ints"]
    close_floats = SimpleClusterSamples.close_join["floats"]
    agg = FormJets.GeneralisedKT((close_ints[:2], close_floats[:2]),
                                 memory_cap=10)
    new_ints, new_floats = agg.combine_ints_floats(0, 1, 0.1**2)
    tst.assert_allclose(new_ints, close_ints[-1])
    tst.assert_allclose(new_floats, close_floats[-1])


def test_Agglomerative_get_decendants():
    """
    Get all decendants of a chosen particle
    within the structure of the jet.

    Parameters
    ----------
    last_only : bool
        Only return the end point decendants
        (Default value = True)
    start_label : int
        start_label used to identify the starting particle
        if not given idx required
        (Default value = None)
    start_idx : int
        Internal index to identify the particle
        if not given start_label required
        (Default value = None)

    Returns
    -------
    decendants : list of ints
        local indices of the decendants

    """
    ints = np.array([[0, -1, -1, -1, 0]])
    floats = np.zeros((1, len(FormJets.Agglomerative.float_columns)))
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    out = agg.get_decendants(True, start_label=0)
    tst.assert_allclose(out, [0])
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1]])
    floats = np.zeros((3, len(FormJets.Agglomerative.float_columns)))
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    out = agg.get_decendants(True, start_label=0)
    assert set(out) == {0}
    out = agg.get_decendants(True, start_label=1)
    assert set(out) == {0, 1}
    out = agg.get_decendants(False, start_label=1)
    assert set(out) == {0, 1, 2}
    out = agg.get_decendants(True, start_idx=2)
    assert set(out) == {0, 1}
    out = agg.get_decendants(False, start_idx=2)
    assert set(out) == {0, 1, 2}


def test_Agglomerative_split():
    """
    Split this jet into as many unconnected jets as it contains

    Returns
    -------
    jet_list : list of Clustering
        the indervidual jets found in here
    """
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1]])
    floats = np.zeros((3, len(FormJets.Agglomerative.float_columns)))
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    jet_list = agg.split()
    assert len(jet_list) == 1
    assert len(jet_list[0].Label) == 3
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1],
                     [3, -1, -1, -1, 0]])
    floats = np.zeros((4, len(FormJets.Agglomerative.float_columns)))
    agg = FormJets.GeneralisedKT((ints, floats), memory_cap=10)
    jet_list = agg.split()
    assert len(jet_list) == 2
    assert {len(j.Label) for j in jet_list} == {1, 3}

#class GeneralisedKT(Agglomerative):

def fill_angular(floats, change_energy=True, energy_boost_factor=1):
    """ given floast that contain px py pz calculate the other values"""
    px, py, pz = floats[4:7]
    if change_energy is True:
        energy = ((1 + energy_boost_factor*np.random.rand()) *
                  np.linalg.norm([px, py, pz], 2))
    else:
        energy = floats[3]
    floats[0] = np.linalg.norm([px, py], 2)
    if energy == pz and px == 0 and py == 0:
        floats[1] = np.Inf
    else:
        floats[1] = 0.5*np.log((energy + pz)/(energy - pz))
    floats[2] = np.arctan2(py, px)
    floats[3] = energy


def test_GeneralisedKT_run():
    """Perform the clustering, without storing debug_data."""
    # shouldn't choke on an empty event
    floats = np.empty((0, 8))
    # need to keep the eventwise file around
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.parquet"))
        set_JetInputs(eventWise, floats)
        eventWise.selected_event = 0
        jet = FormJets.GeneralisedKT(eventWise)
        jet.debug_run()
    n_points = 10
    ints = -np.ones((n_points, len(FormJets.GeneralisedKT.int_columns)),
                    dtype=int)
    ints[:, 0] = np.arange(n_points)
    np.random.seed(1)
    floats = np.random.rand(n_points, len(FormJets.GeneralisedKT.float_columns))*10
    for i in range(n_points):
        fill_angular(floats[i])
    # the aim is to prove that the the first step is not conceptually
    # diferent from latter steps
    params = {"DeltaR": np.inf}
    jet1 = FormJets.GeneralisedKT((ints, floats), dict_jet_params=params)
    jet1.setup_internal()
    idx1, idx2 = jet1.chose_pair()
    jet1.step(idx1, idx2)
    # make a jet out of the ints and floats
    ints2 = jet1._ints[jet1.Label != -1]
    floats2 = jet1._floats[jet1.Label != -1]
    jet2 = FormJets.GeneralisedKT((ints2, floats2), dict_jet_params=params)
    jet1.run()
    jet2.run()
    num_parts = np.sum(jet1.Label != -1)
    tst.assert_allclose(jet1._ints[:num_parts], jet2._ints[:num_parts])
    tst.assert_allclose(jet1._floats[:num_parts], jet2._floats[:num_parts])


def test_GeneralisedKT_simple_cases():
    """Perform the clustering, without storing debug_data."""
    floats = SimpleClusterSamples.two_degenerate['floats']
    ints = SimpleClusterSamples.two_degenerate['ints']
    # for any deltaR, no matter how small, degenerate particles should join
    params = dict(DeltaR=0.001)
    jet1 = FormJets.GeneralisedKT((ints, floats), dict_jet_params=params,
                                  run=True)
    expected_floats = SimpleClusterSamples.degenerate_join['floats']
    expected_ints = SimpleClusterSamples.degenerate_join['ints']
    mask = jet1.Label != -1
    match_ints_floats(expected_ints, expected_floats,
                      jet1._ints[mask], jet1._floats[mask])
    # now try close
    floats = SimpleClusterSamples.two_close['floats']
    ints = SimpleClusterSamples.two_close['ints']
    # for a very small deltaR these don't join
    params = dict(DeltaR=0.001)
    jet1 = FormJets.GeneralisedKT((ints, floats), dict_jet_params=params,
                                  run=True)
    mask = jet1.Label != -1
    match_ints_floats(ints, floats,
                      jet1._ints[mask], jet1._floats[mask])
    # with a more normal deltaR they do
    expected_floats = SimpleClusterSamples.close_join['floats']
    expected_ints = SimpleClusterSamples.close_join['ints']
    params = dict(DeltaR=0.5)
    jet1 = FormJets.GeneralisedKT((ints, floats), dict_jet_params=params,
                                  run=True)
    mask = jet1.Label != -1
    match_ints_floats(expected_ints, expected_floats,
                      jet1._ints[mask], jet1._floats[mask])
    # far appart objects don't tend to join
    floats = SimpleClusterSamples.two_oposite['floats']
    ints = SimpleClusterSamples.two_oposite['ints']
    params = dict(DeltaR=0.5)
    jet1 = FormJets.GeneralisedKT((ints, floats), dict_jet_params=params,
                                  run=True)
    mask = jet1.Label != -1
    match_ints_floats(ints, floats,
                      jet1._ints[mask], jet1._floats[mask])
    

#class Spectral(Agglomerative):

def test_Spectral_run():
    """Perform the clustering, without storing debug_data."""
    n_points = 10
    ints = -np.ones((n_points, len(FormJets.Spectral.int_columns)),
                    dtype=int)
    ints[:, 0] = np.arange(n_points)
    np.random.seed(1)
    floats = np.random.rand(n_points, len(FormJets.Spectral.float_columns))*10
    for i in range(n_points):
        fill_angular(floats[i])
    # the aim is to prove that the the first step is not conceptually
    # diferent from latter steps
    basic = {"MaxMeanDist": np.inf,
            "ExpofPTInput": 1., "ExpofPTAffinity": 1., "ExpofPTEmbedding": 1.}
    def param_run(params):
        jet1 = FormJets.Spectral((ints, floats), dict_jet_params=params)
        jet1.setup_internal()
        idx1, idx2 = jet1.chose_pair()
        jet1.step(idx1, idx2)
        # make a jet out of the ints and floats
        ints2 = jet1._ints[jet1.Label != -1]
        floats2 = jet1._floats[jet1.Label != -1]
        jet2 = FormJets.Spectral((ints2, floats2), dict_jet_params=params)
        jet1.run()
        jet2.run()
        num_parts = np.sum(jet1.Label != -1)
        tst.assert_allclose(jet1._ints[:num_parts], jet2._ints[:num_parts])
        tst.assert_allclose(jet1._floats[:num_parts], jet2._floats[:num_parts])

    for pt_format in FormJets.Spectral.permited_values["ExpofPTFormatInput"]:
        for pt_position in ["Input", "Affinity", "Embedding"]:
            altered = {**basic}
            altered[f"ExpofPTFormat{pt_position}"] = pt_format
            param_run(altered)

    for laplacian in FormJets.Spectral.permited_values["Laplacian"]:
        altered = {**basic}
        altered["Laplacian"] = laplacian
        param_run(altered)

    for hardness in FormJets.Spectral.permited_values["EmbedHardness"]:
        altered = {**basic}
        altered["EmbedHardness"] = hardness
        param_run(altered)


def test_Spectral_embedding_distance2():
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1]])
    floats = np.zeros((3, len(FormJets.Agglomerative.float_columns)))
    params = {"ExpofPTFormatEmbedding": "genkt", "ExpofPTEmbedding": 2,
              "EmbedDistance": 'euclidiean' }
    spect = FormJets.Spectral((ints, floats), memory_cap=10,
                              dict_jet_params=params)
    spect.setup_internal()
    space = np.zeros((3, 2))
    pt = np.arange(3) + 1
    raw_distances, distances = spect._embedding_distance2(space, pt=pt)
    expected = np.zeros((3, 3))
    np.fill_diagonal(expected, np.inf)
    # square roots reduce max accuracy
    tst.assert_allclose(raw_distances, expected, atol=1e-5)
    tst.assert_allclose(distances, expected, atol=1e-5)
    # more complex space
    space = np.arange(3).reshape((3, 1))
    raw_distances, distances = spect._embedding_distance2(space, pt=pt)
    expected = np.array([[np.inf, 1., 4.],
                         [1., np.inf, 16.],
                         [4., 16., np.inf]])
    tst.assert_allclose(expected, raw_distances)
    tst.assert_allclose(expected, distances)
    # try without the kt factor, but with a SingularitySuppression
    params = {"ExpofPTFormatEmbedding": None, "SingularitySuppression": 1,
              "EmbedDistance": 'euclidiean' }
    spect = FormJets.Spectral((ints, floats), memory_cap=10,
                              dict_jet_params=params)
    spect.setup_internal()
    space = np.arange(3).reshape((3, 1))
    pt = np.arange(3) + 0
    singularity = np.array([[1, 0., 0.],
                            [0., 1, 0.5],
                            [0., 0.5, 1]])
    raw_distances, distances = spect._embedding_distance2(space, singularity, pt)
    raw_expected = np.array([[np.inf, 1., 4.],
                             [1., np.inf, 1.],
                             [4., 1., np.inf]])
    expected = np.array([[np.inf, 0., 0.],
                         [0., np.inf, 0.5],
                         [0., 0.5, np.inf]])
    tst.assert_allclose(raw_expected, raw_distances)
    tst.assert_allclose(expected, distances)


def test_Spectral_size():
    ints = np.array([[0, -1, -1, -1, -1],
                     [2, -1, -1, -1, -1]])
    floats = np.zeros((2, len(FormJets.Agglomerative.float_columns)))
    params = {"ExpofPTFormatEmbedding": "genkt", "ExpofPTEmbedding": 2}
    spect = FormJets.Spectral((ints, floats), memory_cap=10,
                              dict_jet_params=params)
    spect.setup_internal()
    expected_inital_size = spect._affinity[spect._2d_avaliable_indices][0, 1]
    tst.assert_allclose(spect.Available_Size, expected_inital_size)
    new_ints, new_floats = spect.combine_ints_floats(0, 1, 0.)
    tst.assert_allclose(new_floats[spect._col_num["Size"]],
                        expected_inital_size*2)


def test_Spectral_stopping_condition():
    """ Will be called before taking another step.

    Parameters
    ----------
    idx1 : int
        index of first of the pair of particles to next join.
    idx2 : int
        index of second of the pair of particles to next join.

    Returns
    -------
    : bool
        True if the clustering should stop now, else False.
    """
    ints = -np.ones((3, len(FormJets.Agglomerative.int_columns)))
    ints[:, 0] = np.arange(3)
    floats = np.zeros((3, len(FormJets.Agglomerative.float_columns)))
    params = {"MaxMeanDist": 10}
    spect = FormJets.Spectral((ints, floats), memory_cap=10,
                              dict_jet_params=params)
    distances2 = np.array([[np.inf, 1., 4.],
                           [1., np.inf, 16.],
                           [4., 16., np.inf]])
    spect._raw_distances2 = distances2
    outcome = spect.stopping_condition(1, 1)
    assert not outcome
    distances2 = np.array([[np.inf, 110., 100.],
                           [110., np.inf, 100.],
                           [100., 100., np.inf]])
    spect._raw_distances2 = distances2
    outcome = spect.stopping_condition(1, 0)
    assert outcome


def test_create_jet_contents():
    """Process the unsplit jets in jet_list to from a content dict for an
    eventWise.

    Parameters
    ----------
    jet_list : list of Agglomerative
        the unsplit agglomerative jets, one for each event.
    existing_contents : dict
        The contents dict for all previous events

    Return
    ------
    contents : dict
        A dict with the contents of previous and new events.
    """
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1]])
    floats = np.zeros((3, len(FormJets.Agglomerative.float_columns)))
    floats[:, 0] = np.arange(3)
    agg1 = FormJets.GeneralisedKT((ints, floats), jet_name="DogJet", memory_cap=10)
    ints = np.array([[0, 1, -1, -1, 0],
                     [2, 1, -1, -1, 0],
                     [1, -1, 2, 0, 1],
                     [3, -1, -1, -1, 0]])
    floats = np.ones((4, len(FormJets.Agglomerative.float_columns)))
    agg2 = FormJets.GeneralisedKT((ints, floats), jet_name="DogJet", memory_cap=10)
    ints = np.empty((0, len(FormJets.Agglomerative.int_columns)))
    floats = np.empty((0, len(FormJets.Agglomerative.float_columns)))
    agg3 = FormJets.GeneralisedKT((ints, floats), jet_name="DogJet", memory_cap=10)
    jet_list = [agg1, agg2, agg3]
    contents = FormJets.create_jet_contents(jet_list, {})
    assert len(contents["DogJet_Label"]) == 3
    assert len(contents["DogJet_Label"][0]) == 1
    assert len(contents["DogJet_Label"][1]) == 2
    assert len(contents["DogJet_Label"][2]) == 0
    tst.assert_allclose(sorted(contents["DogJet_PT"][0][0]), np.arange(3))


def test_get_jet_names_params():
    jet_class = FormJets.Spectral
    jet_paramsA = {'MaxMeanDist': 0.4,
                   'ExpofPTInput': 2.}
    floats = np.random.random((3, 8))
    # set distance to 0
    floats[:, -1] = 0.
    for i in range(3):
        fill_angular(floats[i])
    # need to keep the eventwise file around
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.parquet"))
        set_JetInputs(eventWise, floats)
        eventWise.selected_event = 0
        jet_list = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            eventWise.selected_event = 0
            jets = jet_class(eventWise, dict_jet_params=jet_paramsA,
                             run=True, jet_name="AAJet")
            FormJets.check_params(jets, eventWise)
            jet_list = [jets]
            content = FormJets.create_jet_contents(jet_list, {})
            eventWise.append(**content)
            jet_paramsB = {k: v for k, v in jet_paramsA.items()}
            jet_paramsB['ExpofPTFormatEmbedding'] = None
            eventWise.selected_event = 0
            jets = jet_class(eventWise, dict_jet_params=jet_paramsB,
                             run=True, jet_name="BBJet")
            jet_list = [jets]
            content = FormJets.create_jet_contents(jet_list, {})
            eventWise.append(**content)
            FormJets.check_params(jets, eventWise)
        jet_names = FormJets.get_jet_names(eventWise)
        assert "AAJet" in jet_names
        assert "BBJet" in jet_names
        assert len(jet_names) == 2
        found_paramsA = FormJets.get_jet_params(eventWise, "AAJet",
                                                add_defaults=True)
        for name in jet_paramsA:
            assert found_paramsA[name] == jet_paramsA[name]
        found_paramsB = FormJets.get_jet_params(eventWise, "BBJet",
                                                add_defaults=False)
        for name in jet_paramsB:
            assert found_paramsB[name] == jet_paramsB[name]


def test_check_hyperparameters():
    params1 = {'MaxMeanDist': .2, 'EigenvalueLimit': np.inf,
               'ExpofPTFormatEmbedding': 'genkt', 'ExpofPTInput': 0,
               'Laplacian': 'symmetric',
               'PhyDistance': 'angular'}
    FormJets.check_hyperparameters(FormJets.Spectral, params1)
    params1['MaxMeanDist'] = -1
    with pytest.raises(ValueError):
        FormJets.check_hyperparameters(FormJets.Spectral, params1)
    # GeneralisedKT should not have all these params
    params1['MaxMeanDist'] = 1
    with pytest.raises(ValueError):
        FormJets.check_hyperparameters(FormJets.GeneralisedKT, params1)


def test_filter_jets():
    # will need 
    # Jet_Parent, Jet_Child1, Jet_PT
    min_pt = 0.1
    min_ntracks = 2
    params = {}
    # an empty event should return nothing
    params['Jet_Parent'] = [ak.from_iter([])]
    params['Jet_Child1'] = [ak.from_iter([])]
    params['Jet_PT'] =     [ak.from_iter([])]
    params['MCPID'] =      [ak.from_iter([])]
    params['Generated_mass'] =[ak.from_iter([])]
    # an event with nothing that passes cuts
    params['Jet_Parent'] += [ak.from_iter([[-1], [-1, 1, 1, 2, 2]])]
    params['Jet_Child1'] += [ak.from_iter([[-1], [1, 2, -1, -1, -1]])]
    params['Jet_PT'] +=     [ak.from_iter([[50.,], [0.2, 0.1, 0., 0., .1]])]
    params['MCPID'] +=      [ak.from_iter([25])]
    params['Generated_mass']+=[ak.from_iter([40.])]
    # an event with somthing that passes cuts
    params['Jet_Parent'] += [ak.from_iter([[-1], [-1, 1, 1]])]
    params['Jet_Child1'] += [ak.from_iter([[-1], [1, -1, -1]])]
    params['Jet_PT'] +=     [ak.from_iter([[50.,], [21., 0.1, 0.]])]
    params['MCPID'] +=      [ak.from_iter([25])]
    params['Generated_mass']+=[ak.from_iter([40.])]
    # an event to make use of variable cuts
    params['Jet_Parent'] += [ak.from_iter([[-1, 0], [-1, 1, 1]])]
    params['Jet_Child1'] += [ak.from_iter([[-1, -1], [1, -1, -1]])]
    params['Jet_PT'] +=     [ak.from_iter([[50., 20.], [16., 0.1, 0.]])]
    params['MCPID'] +=      [ak.from_iter([25])]
    params['Generated_mass']+=[ak.from_iter([40.])]
    # an event to make use of variable cuts
    params['Jet_Parent'] += [ak.from_iter([[-1, 0], [-1, 1, 1]])]
    params['Jet_Child1'] += [ak.from_iter([[-1, -1], [1, -1, -1]])]
    params['Jet_PT'] +=     [ak.from_iter([[50., 20.], [14., 0.1, 0.]])]
    params['MCPID'] +=      [ak.from_iter([25])]
    params['Generated_mass']+=[ak.from_iter([40.])]
    # an event to make use of variable cuts
    params['Jet_Parent'] += [ak.from_iter([[-1, 0], [-1, 1, 1]])]
    params['Jet_Child1'] += [ak.from_iter([[-1, -1], [1, -1, -1]])]
    params['Jet_PT'] +=     [ak.from_iter([[17., 20.], [16., 0.1, 0.]])]
    params['MCPID'] +=      [ak.from_iter([25])]
    params['Generated_mass']+=[ak.from_iter([40.])]
    params = {key: ak.from_iter(val) for key,val in params.items()}
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        ew.append(**params)
        # using defaults
        jet_idxs = FormJets.filter_jets(ew, "Jet")
        assert len(jet_idxs[0]) == 0
        assert len(jet_idxs[1]) == 0
        assert len(jet_idxs[2]) == 1
        assert len(jet_idxs[3]) == 2
        assert len(jet_idxs[4]) == 1
        assert len(jet_idxs[5]) == 0
        assert 1 in jet_idxs[2]
        assert 0 in jet_idxs[3]
        assert 1 in jet_idxs[3]
        assert 0 in jet_idxs[4]
        # using selected values
        jet_idxs = FormJets.filter_jets(ew, "Jet", min_pt, min_ntracks)
        assert len(jet_idxs[0]) == 0
        assert len(jet_idxs[1]) == 1
        assert 1 in jet_idxs[1]
        assert len(jet_idxs[2]) == 1
        assert 1 in jet_idxs[2]


def compare_FastJet_FormJets(floats, deltaR, expofPTInput):
    """Helper function, that checks that clustering produced
    by fastjet match the GeneralisedKT answers"""
    # set distance to 0
    floats[:, -1] = 0.
    for i in range(len(floats)):
        fill_angular(floats[i], energy_boost_factor=10)
    # need to keep the eventwise file around
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.parquet"))
        set_JetInputs(eventWise, floats)
        eventWise.selected_event = 0
        dict_jet_params = {"DeltaR": deltaR, "ExpofPTInput": expofPTInput,
                           "ExpofPTFormatInput": "genkt"}
        genkt = FormJets.GeneralisedKT(eventWise,
                                       dict_jet_params=dict_jet_params,
                                       run=True)
        genkt_labels = [set(jet.Leaf_Label) for jet in genkt.split()]
        fastjet = FastJetPython.run_applyfastjet(eventWise, deltaR,
                                                 expofPTInput, "Jet")
        for jet in fastjet.split():
            labels = set(jet.Leaf_Label)
            assert labels in genkt_labels, f"{labels}, not in {genkt_labels}"


def test_compare_FastJet_FormJets():
    try:
        FastJetPython.compile_fastjet()
    except Exception:
        return
    floats = SimpleClusterSamples.two_degenerate["floats"]
    compare_FastJet_FormJets(floats, 0.4, -1)
    compare_FastJet_FormJets(floats, 0.4, 0)
    compare_FastJet_FormJets(floats, 0.4, 1)

    floats = SimpleClusterSamples.two_close["floats"]
    compare_FastJet_FormJets(floats, 0.4, -1)
    compare_FastJet_FormJets(floats, 0.4, 0)
    compare_FastJet_FormJets(floats, 0.4, 1)

    floats = SimpleClusterSamples.two_oposite["floats"]
    compare_FastJet_FormJets(floats, 0.4, -1)
    compare_FastJet_FormJets(floats, 0.4, 0)
    compare_FastJet_FormJets(floats, 0.4, 1)

    num_random_trials = 100
    permitted_mistakes = 300
    found_mistakes = [0, 0, 0]
    num_particles = 3
    for i in range(num_random_trials):
        floats = np.random.rand(num_particles,
            len(FormJets.GeneralisedKT.float_columns))*10
        for exp in [-1, 0, 1]:
            try:
                compare_FastJet_FormJets(floats, 0.4, exp)
            except AssertionError as e:
                #print(f"Missmatch with fastjet, q={exp};\n{floats}")
                print(f"Missmatch with fastjet, q={exp}")
                found_mistakes[exp+1] += 1
                if np.sum(found_mistakes) > permitted_mistakes:
                    raise e
    return found_mistakes

