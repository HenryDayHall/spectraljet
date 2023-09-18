import sys
import os
sys.path.append('sgwt')

import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import bisect
from . import Components, Constants
from sgwt import sgwt_functions 
import scipy.spatial
import scipy.linalg
import awkward as ak


def knn(distances, num_neighbours):
    order = np.argsort(np.argsort(distances, axis=0), axis=0)
    neighbours = order <= num_neighbours
    # the distances obect can be 1d, in which case we are done
    # or, we need the transpose becuase it should be symmetric
    try:
        # adds a transpose of neighbours to neighbours
        neighbours = neighbours + list(map(list, zip(*neighbours)))
    except TypeError:
        pass  # it was 1d
    neighbours = neighbours.astype(bool)
    return neighbours


class Custom_KMeans:
    """ Compute kmeans with a custom distance function,
    the mean of the centroids is a euclidien mean, but the
    scores and allocations are done with the given  distance function"""
    def __init__(self, distance_function, max_iter=500, max_restarts=500, silent=True):
        self.distance_function = distance_function
        self.silent = silent
        self.max_iter = max_iter
        self.max_restarts = max_restarts

    def fit(self, n_clusters, points):
        if len(points) < n_clusters:
            raise ValueError("Cannot make more clusters than there are points")
        if n_clusters < 1:
            raise ValueError("Cannot make less that 1 cluster")
        self.points = points
        self.n_points = len(points)
        min_score = np.inf
        min_count = 0
        patience = 5
        scores = []
        for _ in range(self.max_restarts):
            score, aloc, cen = self._attempt(n_clusters)
            if np.isclose(score, min_score):
                # the elif condition below will always be hit first
                allocations.append(aloc)
                centeroids.append(cen)
                scores.append(score)
                min_count += 1
                if min_count > patience:
                    break
            elif score < min_score:
                allocations = [aloc]
                centeroids = [cen]
                scores = [score]
                min_count = 0
                min_score = score
        else:
            if not self.silent:
                print("Didn't repeat.")
            # if all scores were nan we could technically reach here
            # with no scores list
            if not scores:
                # return the last one
                return score, aloc, cen
        # it is unlikely but possible that there are multiple choices
        best = np.argmin(scores)
        return scores[best], allocations[best], centeroids[best]


    def _attempt(self, n_clusters):
        centeroids = self._inital_centroids(n_clusters)
        allocations = -np.ones(self.n_points)
        for i in range(self.max_iter):
            distances = self.distance_function(centeroids, self.points, None, None)
            # sometimes a point will return all nan, for an angular function
            # this is a point sitting at the origin
            # it dosen't matter where we put this point
            distances[:, np.all(np.isnan(distances), axis=0)] = -1
            # other nan values should be respected
            new_allocations = np.nanargmin(distances, axis=0)
            if np.all(new_allocations == allocations):
                break
            allocations = new_allocations
            for cluster_n in range(n_clusters):
                points_here = self.points[allocations==cluster_n]
                if len(points_here):  # otherwise leave the centeriod where it was
                    centeroid = np.nanmean(points_here, axis=0)
                    if not np.all(centeroid == 0):
                        # all 0 centeroid breaks the cross product thing,
                        # and just indicates that the points in this cluster
                        # are orientated in oposing directions
                        centeroids[cluster_n] = centeroid
        else:
            if not self.silent:
                print("Didn't settle!!")
        score = np.nansum(distances[allocations, range(self.n_points)])
        return score, allocations, centeroids


    def _inital_centroids(self, n_clusters):
        indices = np.random.choice(self.n_points, n_clusters, replace=False)
        return self.points[indices]


def parity(identities_a, coordinates_a, identities_b, coordinates_b):
    """ Parity for b such that is best matches a """
    n_dims = min(len(coordinates_a[0]), len(coordinates_b[0]))
    assert len(identities_a) > len(identities_b)
    identities_a = list(identities_a)
    indices_in_b = np.where([b in identities_a for b in identities_b])[0]
    indices_in_a = [identities_a.index(b) for b in identities_b[indices_in_b]]
    coordinates_a = coordinates_a[indices_in_a, :n_dims]
    coordinates_b = coordinates_b[indices_in_b, :n_dims]
    diff = np.sum(np.abs(coordinates_a - coordinates_b), axis=0)
    flipped_diff = np.sum(np.abs(coordinates_a + coordinates_b), axis=0)
    par = 1 - 2*(flipped_diff < diff)
    return par


def return_one(*args, **kwargs):
    """Used to make a no-op"""
    return 1


def ca_distances2(rapidity, phi, rapidity_column=None, phi_column=None):
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
    if rapidity_column is None:
        rapidity_column = np.expand_dims(rapidity, 1)
    rapidity_distances = np.abs(rapidity - rapidity_column)
    if phi_column is None:
        phi_column = np.expand_dims(phi, 1)
    phi_distances = Components.angular_distance(phi, phi_column)
    distances2 = phi_distances**2 + rapidity_distances**2
    return distances2


def genkt_factor(exponent, pt, pt_column=None):
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
    pt_power = pt**exponent
    if pt_column is None:
        pt_power_column = np.expand_dims(pt_power, 1)
    else:
        pt_power_column = pt_column**exponent
    factor = np.minimum(pt_power, pt_power_column)
    return factor


def ratiokt_factor(exponent, pt, pt_column=None):
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
    pt_power = pt**exponent
    if pt_column is None:
        pt_power_column = np.expand_dims(pt_power, 1)
    else:
        pt_power_column = pt_column**exponent
    mins = np.minimum(pt_power, pt_power_column)
    maxes = np.maximum(pt_power, pt_power_column)
    ratio = mins/maxes
    return ratio


def hyperbolic_suppression(constant, values):
    return 1 - (constant/(constant + values))


def suppressedkt_factor(constant, pt, pt_column=None):
    """A suppresed kt factor, which is small when the minimum pt is small

    Parameters
    ----------
    constant : float
        the constant above which suppression stops
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
    if pt_column is None:
        pt_column = np.expand_dims(pt, 1)
    mins = np.minimum(pt, pt_column)
    factor = hyperbolic_suppression(constant, mins)
    return factor


def singularity_factor(constant, angular_distance2_matrix, pt, pt_column=None,
                       diagonal_index=None):
    """ Calculate how singular each point is"""
    kt_factor2 = genkt_factor(2, pt, pt_column)
    combined_distance = np.sqrt(kt_factor2*angular_distance2_matrix)
    factor = hyperbolic_suppression(constant, combined_distance)
    if pt_column is None:
        np.fill_diagonal(factor, 1)
    else:
        factor[diagonal_index] = 1
    return factor


def exp_affinity(distances2, sigma=1, exponent=2, fill_diagonal=True):
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
    aff = np.exp(-(distances2**(0.5*exponent))/sigma)
    if fill_diagonal:
        np.fill_diagonal(aff, 0.)
    return aff


def unnormed_laplacian(affinities, _=None):
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
    diagonal = np.diag(np.sum(affinities, axis=-2))
    laplacian = diagonal - affinities
    return laplacian


def normalised_laplacian(affinities, norm_by):
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
    laplacian = unnormed_laplacian(affinities)
    if len(norm_by.shape) < len(affinities.shape):  # it's a 1d list of weights
        norm_by_invhalf = np.diag(1/np.sqrt(norm_by))
    else:
        # need to do a real matrix inverse
        norm_by_invhalf = np.linalg.inv(np.sqrt(norm_by))
    laplacian = np.matmul(norm_by_invhalf, np.matmul(laplacian,
                                                     norm_by_invhalf))
    # if the affinity contians disconeccted particles,
    # there will be nan in the laplacian
    np.nan_to_num(laplacian, copy=False)
    return laplacian


def symmetric_laplacian(affinities, _=None):
    """Construct a symmetric laplacian, L = D^-1/2(D-A)D^-1/2

    Parameters
    -------
    affinities : 2d array of float
        Square grid of affinities between points.
    _ : Object
        optional second parameter for iterface consistany
        
    Returns
    -------
    laplacian : 2d array of float
        Laplacian of the graph.
    """
    norm_by = np.sum(affinities, axis=-2)
    return normalised_laplacian(affinities, norm_by)


def embedding_space(laplacian, num_dims=None, max_eigval=None):
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
    if len(laplacian) == 0:
        return np.empty(0), np.empty((0, 0))
    if num_dims is None and max_eigval is None:
        raise ValueError
    if max_eigval is not None:
        values, vectors = scipy.linalg.eigh(laplacian,
                                            subset_by_value=(-np.inf, max_eigval))
        values = values[1:]
        vectors = vectors[:, 1:]
        #values, vectors = scipy.linalg.eigh(laplacian)
        #keep = values < max_eigval
        #keep[0] = False
        #values = values[keep]
        #vectors = vectors[:, keep]
    else:
        values, vectors = scipy.linalg.eigh(laplacian, eigvals=(1, num_dims))
    return values, vectors

    
def dimension_hardness(embedding, pt):
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
    pt_column = np.expand_dims(pt, 1)
    hardness = np.mean(np.abs(embedding)*pt_column, axis=0)
    return np.expand_dims(hardness, -2)


def dimension_scaling(eigenvalues, beta, clip_beta):
    """Calculate the scaling factors of each dimensions of the embedding space.
    Used to ensure dimensions with lower eigenvalues are larger.

    Parameters
    ----------
    eigenvalues : array of floats
        eigenvalues of the embedding space
    beta : float
        power to raise the eigenvectors to
    clip_beta: float
        lowest value to clip the eigenvalues to

    Returns
    -------
    scale : array of floats
        the scaling for each dimension
    """
    scale = np.clip(eigenvalues, clip_beta, None)**(-beta)
    return np.expand_dims(scale, -2)


def embedding_norm(embedding):
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
    assert len(embedding.shape) == 2
    norm = np.expand_dims(np.sqrt(np.sum(embedding**2, axis=-1)), -1)
    embedding = embedding/norm
    return embedding


def embedding_distance2(embedding, embedding_column=None):
    """Distances squared in the embedding space.

    Parameters
    ----------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space
    embedding_column : 2d array of float (optional)
        Column of embedding values.
        If not given, taken as the transpose of the row.

    Returns
    -------
    distances2 : (n, n) array of floats
        distances between particles squared
    """
    if embedding_column is None:
        distances2 = scipy.spatial.distance.pdist(embedding, metric='sqeuclidean')
        distances2 = scipy.spatial.distance.squareform(distances2)
        np.fill_diagonal(distances2, np.inf)
    else:
        distances2 = scipy.spatial.distance.cdist(embedding, embedding_column,
                                                  metric='sqeuclidean')
    return distances2


def embedding_angular2(embedding, embedding_column=None):
    """Angular distances squared in the embedding space.

    Parameters
    ----------
    embedding : (n, m) array of floats
        positions of the particles in the embediding space
    embedding_column : 2d array of float (optional)
        Column of embedding values.
        If not given, taken as the transpose of the row.

    Returns
    -------
    distances2 : (n, n) array of floats
        distances between particles squared
    """
    if embedding_column is None:
        distances2 = scipy.spatial.distance.pdist(embedding, metric='cosine')
        distances2 = scipy.spatial.distance.squareform(distances2)
    else:
        distances2 = scipy.spatial.distance.cdist(embedding, embedding_column,
                                                  metric='cosine')
    distances2 = np.arccos(1 - distances2)**2
    if embedding_column is None:
        np.fill_diagonal(distances2, np.inf)
    return distances2


def embedding_root_angular2(embedding, embedding_column=None):
    distances2 = embedding_angular2(embedding, embedding_column)
    return np.sqrt(distances2)


def mean_distance(distance2_matrix, limit):
    """ Compare the mean of a symmetric matrix of distances
    against a specified limit,
    ignores diagonal entries

    Parameters
    ----------
    distance2_matrix : array like
        square matrix of distances squared
    limit : float
        value to compare with

    Returns
    -------
    mean_distance : float
        mean of the distances
    """
    distance2 = distance2_matrix[np.tril_indices_from(distance2_matrix, -1)]
    return np.mean(np.sqrt(distance2)) < limit


def min_distance(distance2_matrix, limit):
    """ Compare the min of a symmetric matrix of distances
    against a specified limit, ignores diagonal entries

    Parameters
    ----------
    distance2_matrix : array like
        square matrix of distances squared
    limit : float
        value to compare with

    Returns
    -------
    mean_distance : float
        mean of the distances
    """
    distance2 = distance2_matrix[np.tril_indices_from(distance2_matrix, -1)]
    return np.sqrt(np.min(distance2)) < limit


class Clustering:
    int_columns = ["Label",
                   "Parent", "Child1", "Child2",
                   "Rank"]
    float_columns = ["PT", "Rapidity", "Phi",
                     "Energy", "Px", "Py", "Pz",
                     "JoinDistance", "Size"]
    default_params = {}
    permited_values = {}
    debug_attributes = ["avaliable_mask"]

    def __init__(self, *args, **kwargs):
        """
        Class constructor

        Parameters
        ----------
        input_data : EventWise or (2d array of ints, 2d array of floats)
            data file for inputs
        jet_name : string (optional)
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        run : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        memory_cap: int (optional)
            Size of largest array to be created during processing.
            In debug modes this will limit how many steps can be retained.
            (Default; 20000)
        """
        # check the args are what tey should be
        permitted_kwargs = ['input_data', 'jet_name', 'dict_jet_params',
                            'run', 'memory_cap']
        unwanted = [name for name in kwargs if name not in permitted_kwargs]
        if unwanted:
            raise ValueError(f"Don't recognise args {unwanted}")
        if 'input_data' not in kwargs:
            assert len(args) == 1, "Only arg should be input_data"
            input_data = args[0]
        else:
            assert len(args) == 0, "Only arg should be input_data"
            input_data = kwargs['input_data']

        # store them as function attributes
        self.memory_cap = kwargs.get('memory_cap', 20000)
        self.setup_hyperparams(kwargs.get('dict_jet_params', {}))
        self.setup_column_numbers()
        self.jet_name = kwargs.get('jet_name', "AgglomerativeJet")
        if not self.jet_name.endswith("Jet"):
            self.jet_name += "Jet"
        self._ints, self._floats = None, None
        self.setup_ints_floats(input_data)
        #  This is called before run;
        #  self.setup_internal()
        if kwargs.get('run', False):
            self.run()

    @classmethod
    def from_kinematics(cls, energy, px, py, pz,
                        pt=None, rapidity=None, phi=None,
                        **kwargs):
        """Alternative constructor, takes the kinematics of the particles.
        
        Parameters
        ----------
        energy : array of floats
            energy of the input particles
        px : array of floats
            px of the input particles
        py : array of floats
            py of the input particles
        pz : array of floats
            pz of the input particles
        pt : array of floats (optional)
            pt of the input particles
        rapidity : array of floats (optional)
            rapidity of the input particles
        phi : array of floats (optional)
            phi of the input particles
        jet_name : string (optional)
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        run : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        memory_cap: int (optional)
            Size of largest array to be created during processing.
            In debug modes this will limit how many steps can be retained.
            (Default; 20000)
        """
        if phi is None or pt is None:
            phi, pt = Components.pxpy_to_phipt(px, py)
        if rapidity is None:
            rapidity = Components.ptpze_to_rapidity(pt, pz, energy)
        n_inputs = len(energy)
        ints = [[i, -1, -1, -1, -1] for i in range(n_inputs)]
        floats = [pt, rapidity, phi, energy, px, py, pz,
                  np.zeros(n_inputs),  # Join distance
                  np.ones(n_inputs)]  # Size
        # transpose a list of iterables
        floats = list(map(list, zip(*floats)))
        return cls((ints, floats), **kwargs)

    def setup_ints_floats(self, input_data):
        """ Create the _ints and _floats, along with
        the _avaliable_mask and _avaliable_idxs

        Parameters
        ----------
        input_data : EventWise or (2d array of ints, 2d array of floats)
            data file for inputs
        """
        if isinstance(input_data, tuple):
            start_ints, start_floats = input_data
        else:  # should be an EventWise
            start_ints, start_floats = self.read_ints_floats(input_data)
        self._ints, self._floats = \
            self.create_int_float_tables(start_ints, start_floats)
        self._avaliable_mask = (self.Label != -1)*(self.Parent == -1)
        self._avaliable_idxs = np.where(self._avaliable_mask)[0].tolist()

    def read_ints_floats(self, eventWise):
        """ Read the data for clustering from a file.

        Parameters
        ----------
        eventWise : EventWise
            data file for inputs

        Returns
        -------
        ints : list of list of int
            initial integer input data for clustering
        floats : list of list of floats
            initial float input data for clustering
        """
        assert eventWise.selected_event is not None
        n_inputs = len(eventWise.JetInputs_PT)
        ints = [[i, -1, -1, -1, -1] for i in range(n_inputs)]
        floats = [eventWise.JetInputs_PT,
                  eventWise.JetInputs_Rapidity,
                  eventWise.JetInputs_Phi,
                  eventWise.JetInputs_Energy,
                  eventWise.JetInputs_Px,
                  eventWise.JetInputs_Py,
                  eventWise.JetInputs_Pz,
                  np.zeros(n_inputs),  # Join distance
                  np.ones(n_inputs)]  # Size
        # transpose a list of iterables
        floats = list(map(list, zip(*floats)))
        return ints, floats

    def create_int_float_tables(self, start_ints, start_floats):
        """Space needed differes between agglomerative and divisive methods

        Parameters
        ----------
        start_ints : list of list of int
            initial integer input data for clustering
        start_floats : list of list of floats
            initial float input data for clustering

        Returns
        -------
        ints : list of list of int
            integer input data for clustering
        floats : list of list of floats
            float input data for clustering
        """
        raise NotImplementedError

    def _next_free_row(self):
        """Find the next free index to place a new point.

        Returns
        -------
        i : int
            index of free point
        """
        label = self.Label
        i = next((i for i in range(self.memory_cap)
                  if label[i] == -1), self.memory_cap)
        return i

    def _update_avalible(self, idxs_out, idxs_in=set()):
        """Update which indices are avalible

        Parameters
        ----------
        idxs_out : iterable of ints
            the indices of points that are no longer avaliable.
        idxs_in : iterable of ints (optional)
            the indices of points that are now avaliable.
        """
        for idx in idxs_out:
            #assert self._avaliable_mask[idx]
            self._avaliable_mask[idx] = False
            self._avaliable_idxs.remove(idx)
        for idx in idxs_in:
            #assert not self._avaliable_mask[idx]
            self._avaliable_mask[idx] = True
            bisect.insort(self._avaliable_idxs, idx)
        
    def setup_hyperparams(self, dict_jet_params):
        """
        Using the default parameters and the chosen parameters set
        the attributes of the Clustering to contain the parameters
        used for clustering.  Harmless to call multiple times.

        Parameters
        ----------
        default_params : dict of params
            dictionary of default settings, to be applied
            when parameters are not set elsewhere
            key is parameter name, value is parameter value
        dict_jet_params : dict of params
            parameters may be supplied together as a dictionary
            key is parameter name, value is parameter value
        """
        check_hyperparameters(type(self), dict_jet_params)
        for name in self.default_params:
            value = dict_jet_params.get(name, self.default_params[name])
            setattr(self, name, value)
        # doing it now garantees the hyperparameters are present and correct
        self._setup_clustering_functions()

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        raise NotImplementedError

    def setup_column_numbers(self):
        """
        Using the list of column names make a dict for quick indexing
        """
        self._col_num = {}
        # int columns
        for i, name in enumerate(self.int_columns):
            self._col_num[name] = i
        # float columns
        self._float_contents = {}
        for i, name in enumerate(self.float_columns):
            self._col_num[name] = i

    def __dir__(self):
        """ Ensure the attributes are displayed in consistant order """
        new_attrs = set(super().__dir__())
        columns = self.float_columns + self.int_columns
        new_attrs.update(columns)
        new_attrs.update(["Leaf_" + name for name in columns])
        new_attrs.update(["Available_" + name for name in columns])
        return sorted(new_attrs)

    def __getattr__(self, attr_name):
        """
        Make the attributes for the floats and ints used to construct jets.
        The integer columns are simply returned as numpy arrays.
        """
        if attr_name.startswith("Leaf_"):
            mask = ((self._ints[:, self._col_num["Child1"]] == -1) *
                    (self._ints[:, self._col_num["Label"]] != -1))
            return getattr(self, attr_name[5:])[mask]
        if attr_name.startswith("Available_"):
            return getattr(self, attr_name[10:])[self._avaliable_idxs]
        if attr_name in self.float_columns:
            col_num = self._col_num[attr_name]
            if attr_name == 'Phi':  # make sure it's -pi to pi
                self._floats[:, col_num] = \
                    Components.confine_angle(self._floats[:, col_num])
            values = self._floats[:, col_num]
            return values
        if attr_name in self.int_columns:
            # ints return every value
            col_num = self._col_num[attr_name]
            return self._ints[:, col_num]
        if attr_name == "Rapidity":
            # if the jet was constructed with pseudorapidity
            # we might still want to know the rapidity
            return Components.ptpze_to_rapidity(self.PT, self.Pz, self.Energy)
        if attr_name == "Pseudorapidity":
            # vice verca
            return Components.theta_to_pseudorapidity(self.Theta)
        raise AttributeError(
                f"{self.__class__.__name__} does not have {attr_name}")

    def setup_internal(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Also create all debug attributes."""
        raise NotImplementedError

    def run(self):
        """Perform the clustering, without storing debug_data."""
        raise NotImplementedError

    def debug_run(self):
        """Perform the clustering, storing debug_data."""
        raise NotImplementedError

    def setup_matrix_plt(self, ax):
        """ Setup an axis to plot a matrix

        Parameters
        ----------
        ax: matplotlib.pyplt.Axes
            the axes to plot on
            
        Returns
        ----------
        ax: matplotlib.pyplt.Axes
            the axes to plot on
        """
        if ax is None:
            ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        return ax

    def make_leaf_colours(self):
        """Create a list of np colour objects for each leaf

        Returns
        -------
        colours : dict
            dict of the form {particle index : colour}
        """
        leaf_idxs = np.where((self.Label != -1)*(self.Child1 == -1))[0]
        colour_map = plt.get_cmap('nipy_spectral')
        colours = {idx: colour for idx, colour in 
                   zip(leaf_idxs,
                       colour_map(np.linspace(0.1, 0.9, len(leaf_idxs))))}
        return colours

    def plt_phys_space(self, step, ax=None):
        """ Plot the particles on the barrel.

        Parameters
        ----------
        step : int
            index of step starting from oldest retained step
            which may not be first step
        ax: matplotlib.pyplt.Axes (optional)
            the axes to plot on
        """
        mask = self.debug_data["avaliable_mask"][step]
        colours = self.make_leaf_colours()
        if ax is None:
            ax = plt.gca()
        for idx in np.where(mask)[0]:
            descendant_idxs = self.get_decendants(last_only=True,
                                                  start_idx=idx)
            colour = colours[descendant_idxs[0]]
            rapidity = self.Rapidity[descendant_idxs]
            phi = self.Phi[descendant_idxs]
            size = 3 + np.sqrt(self.PT[descendant_idxs])
            ax.scatter(rapidity, phi, size, color=colour)
        ax.set_xlabel("Rapidity")
        ax.set_ylabel("$\\phi$")
        ax.set_title("Physical space")

    def split(self):
        """
        Split this jet into as many unconnected jets as it contains
        Differs between aglomerative and divisive algorithms.

        Returns
        -------
        jet_list : list of Clustering
            the indervidual jets found in here
        """
        raise NotImplementedError


class Agglomerative(Clustering):
    debug_attributes = ["avaliable_mask"]

    def create_int_float_tables(self, start_ints, start_floats):
        """ Format the data for clustering, allocating memory.
        The tables created have space for pesudojets that will be created.

        Parameters
        ----------
        start_ints : list of list of int
            initial integer input data for clustering
        start_floats : list of list of floats
            initial float input data for clustering

        Returns
        -------
        ints : list of list of int
            integer input data for clustering
        floats : list of list of floats
            float input data for clustering
        """
        start_labels = [row[self._col_num["Label"]] for row in start_ints]
        assert -1 not in start_labels, "-1 is a reserved label"
        # this will form a binary tree,
        # so at most there can be
        # n_inputs + (n_unclustered*(n_unclustered-1))/2
        n_inputs = len(start_ints)
        if n_inputs == 0:
            ints = np.empty((0, len(self.int_columns)))
            floats = np.empty((0, len(self.float_columns)))
            return ints, floats
        assert n_inputs < self.memory_cap, \
            f"More particles ({n_inputs}) than possible " +\
            f"with this memory_cap ({self.memory_cap})"
        # don't assume the form of start_ints
        n_unclustered = np.sum([row[self._col_num["Parent"]] == -1
                                for row in start_ints])
        max_elements = n_inputs + int(0.5*(n_unclustered*(n_unclustered-1)))
        # we limit the maximum elements in memory
        max_elements = min(max_elements, self.memory_cap)
        ints = -np.ones((max_elements, len(self.int_columns)),
                        dtype=int)
        ints[:n_inputs] = start_ints
        floats = np.full((max_elements, len(self.float_columns)),
                         np.nan, dtype=float)
        floats[:n_inputs] = start_floats
        return ints, floats

    @property
    def _2d_avaliable_indices(self):
        """
        Using the _avaliable_idxs make indices for indexing
        the corrisponding minor or a 2d matrix.

        Returns
        -------
        : tuple of arrays
            tuple that will index the matrix minor
        """
        num_avail = len(self._avaliable_idxs)
        avail = np.tile(self._avaliable_idxs, (num_avail, 1)).astype(int)
        return avail.T, avail
        
    def _reoptimise_preallocated(self):
        """Rearange the objects in memory to accomidate more.

        Memory limit has been reached, the preallocated arrays
        need to be rearanged to allow for removing objects which
        are no longer needed.
        anything still in _avaliable_idxs will not be moved.
        Also, remove anything in debug_data, becuase it will be
        invalidated.
        """
        not_avaliable = np.where(~self._avaliable_mask)[0]
        not_a_ints = self._ints[not_avaliable]
        not_a_floats = self._floats[not_avaliable]
        # before they are erased, add them to the back of the arrays
        self._ints = np.concatenate((self._ints, not_a_ints))
        self._floats = np.concatenate((self._floats, not_a_floats))
        # now wipe those rows
        self._ints[not_avaliable] = -1
        self._floats[not_avaliable] = 0.
        # avaliability remains as before
        # so any other
        # matrix can be ignored, assuming you only use
        # avaliable row/cols
        if hasattr(self, "debug_data"):
            # this is outdated, so scrap it
            self.debug_data = {name: [] for name in self.debug_data}

    def run(self):
        """Perform the clustering, without storing debug_data."""
        if not self._avaliable_idxs:
            return
        self.setup_internal()
        while True:
            idx1, idx2 = self.chose_pair()
            if self.stopping_condition(idx1, idx2):
                break
            self.step(idx1, idx2)

    def debug_run(self):
        """Perform the clustering, storing debug_data."""
        self.setup_internal()
        self.debug_data = {name: [np.copy(getattr(self, '_' + name))]
                           for name in self.debug_attributes}
        while self._avaliable_idxs:
            idx1, idx2 = self.chose_pair()
            if self.stopping_condition(idx1, idx2):
                break
            self.step(idx1, idx2)
            for name in self.debug_data:
                self.debug_data[name].append(
                    np.copy(getattr(self, '_' + name)))

    def get_historic_2d_mask(self, step):
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
        mask = self.debug_data["avaliable_mask"][step]
        idxs = np.where(mask)[0]
        num_avail = idxs.shape[0]
        avail = np.tile(idxs, (num_avail, 1))
        return (avail.T, avail)

    def plt_distances(self, step, ax=None):
        """ Plot the distances in the final space.

        Parameters
        ----------
        step : int
            index of step starting from oldest retained step
            which may not be first step
        ax: matplotlib.pyplt.Axes (optional)
            the axes to plot on
        """
        mask_2d = self.get_historic_2d_mask(step)
        ax = self.setup_matrix_plt(ax)
        distances = np.sqrt(self._distances2[mask_2d])
        image = ax.imshow(distances)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Distance")

    def stopping_condition(self, idx1, idx2):
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
        raise NotImplementedError

    def step(self, idx1, idx2):
        """ Perform a step of clustering

        Parameters
        ----------
        idx1 : int
            index of first of the pair of particles to next join.
        idx2 : int
            index of second of the pair of particles to next join.
        """
        if idx1 == idx2:
            self._step_same(idx1)
        else:
            self._step_differ(idx1, idx2)

    def _step_same(self, idx):
        """ Perform a step of clustering, if next closest particle
        is close to the beam.

        Parameters
        ----------
        idx1 : int
            index of the particle near the beam.
        """
        self._update_avalible([idx])

    def _step_differ(self, idx1, idx2):
        """ Perform a step of clustering, if next closest particle
        is close to another particle.

        Parameters
        ----------
        idx1 : int
            index of first of the pair of particles to next join.
        idx2 : int
            index of second of the pair of particles to next join.
        """
        distance2 = self._distances2[idx1, idx2]
        new_int_row, new_float_row = self.combine_ints_floats(idx1, idx2,
                                                              distance2)
        idx_parent = self._next_free_row()
        if idx_parent >= self.memory_cap:
            # we ran out of space
            # at this time, idx1 and idx2 are still marked as avalible
            # so they will not be moved
            self._reoptimise_preallocated()
            # the next free row is now somewhere else
            idx_parent = self._next_free_row()
        self._ints[idx_parent] = new_int_row
        self._floats[idx_parent] = new_float_row
        self._update_avalible([idx1, idx2], [idx_parent])
        if np.sum(self._avaliable_mask) > 1:
            self.update_after_join(idx1, idx2, idx_parent)

    def chose_pair(self):
        """ Find the next two particles to join.

        Return
        ----------
        row : int
            index of first of the pair of particles to next join.
        column : int
            index of second of the pair of particles to next join.
        """
        masked_distances2 = self._distances2[self._2d_avaliable_indices]
        # this is the row and col in the masked array
        row, column = np.unravel_index(np.argmin(masked_distances2),
                                       masked_distances2.shape)
        # this is the row and col in the whole array
        row = self._avaliable_idxs[row]
        column = self._avaliable_idxs[column]
        return row, column

    def combine_ints_floats(self, idx1, idx2, distance2):
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
        new_id = np.max(self.Label) + 1
        self.Parent[idx1] = new_id
        self.Parent[idx2] = new_id
        rank = max(self.Rank[idx1], self.Rank[idx2]) + 1
        # inputidx, parent, idx1, idx2 rank
        # idx1 shoul
        ints = [new_id,
                -1,
                self.Label[idx1],
                self.Label[idx2],
                rank]
        # PT px py pz eta phi energy join_distance
        # it's easier conceptually to calculate pt, phi and rapidity
        # afresh than derive them
        # from the exisiting pt, phis and rapidity
        floats = self._floats[idx1] + self._floats[idx2]
        px = floats[self._col_num["Px"]]
        py = floats[self._col_num["Py"]]
        pz = floats[self._col_num["Pz"]]
        energy = floats[self._col_num["Energy"]]
        phi, pt = Components.pxpy_to_phipt(px, py)
        floats[self._col_num["PT"]] = pt
        floats[self._col_num["Phi"]] = phi
        floats[self._col_num["Rapidity"]] = \
            Components.ptpze_to_rapidity(pt, pz, energy)
        # fix the distance
        floats[self._col_num["JoinDistance"]] = np.sqrt(distance2)
        return ints, floats

    def update_after_join(self, idx1, idx2, idx_parent):
        """Peform updates to internal data, after combining two particles.

        Parameters
        ----------
        idx1 : int
            index of first input particle
        idx2 : int
            index of second input particle
        idx_parent : int
            index of the new particle created
        """
        raise NotImplementedError

    def get_decendants(self, last_only=True, start_label=None, start_idx=None):
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
        label = self.Label
        child1 = self.Child1
        child2 = self.Child2
        local_obs = np.where((label != -1)*(child1 == -1))[0]
        if start_idx is None:
            start_idx = np.where(label == start_label)[0][0]
        elif start_label is None:
            start_label = label[start_idx]
        else:
            raise TypeError("Need to specify a pseudojet")
        label = label.tolist()
        stack = [start_idx]
        decendants = []
        while stack:
            idx = stack.pop()
            if idx in local_obs:
                decendants.append(idx)
                continue
            stack.append(label.index(child1[idx]))
            stack.append(label.index(child2[idx]))
            if not last_only:
                decendants.append(idx)
        return decendants

    def split(self):
        """
        Split this jet into as many unconnected jets as it contains

        Returns
        -------
        jet_list : list of Clustering
            the indervidual jets found in here
        """
        if self._ints.shape[0] == 0:  # nothing else to do if the jet is empty
            return []
        mask = self.Label != -1
        roots = np.where(mask)[0][self.Parent[mask] == -1]
        jet_list = []
        jet_params = {name: getattr(self, name, self.default_params[name])
                      for name in self.default_params}
        for root in roots:
            group = self.get_decendants(last_only=False, start_idx=root)
            ints = self._ints[group]
            floats = self._floats[group]
            # setting the momory cap to the length of the ints
            # prevents any additional space, beyond what is
            # strictly needed being allocated to this jet
            jet = type(self)(input_data=(ints, floats),
                             jet_name=self.jet_name,
                             dict_jet_params=jet_params,
                             memory_cap=len(ints)+1)
            jet_list.append(jet)
        return jet_list


class GeneralisedKT(Agglomerative):
    default_params = {'DeltaR': .8,
                      'ExpofPTInput': 0,
                      'PhyDistance': 'angular',
                      'ExpofPTFormatInput': None,
                      }
    permited_values = {'DeltaR': Constants.numeric_classes['pdn'],
                       'ExpofPTInput': Constants.numeric_classes['rn'],
                       'PhyDistance': ['angular'],
                       'ExpofPTFormatInput': [None, 'genkt', 'ratiokt'],
                       }

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.PhyDistance == 'angular':
            self._calc_distances2 = ca_distances2
        else:
            raise NotImplementedError

        if self.ExpofPTFormatInput is None:
            self._calc_input_pt = return_one
        elif self.ExpofPTFormatInput == 'genkt':
            self._calc_input_pt = genkt_factor
        elif self.ExpofPTFormatInput == 'ratiokt':
            self._calc_input_pt = ratiokt_factor
        else:
            raise NotImplementedError

    def setup_internal(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Also create all debug attributes."""
        kt_factor2 = self._calc_input_pt(self.ExpofPTInput, self.Available_PT)**2
        angular_distances2 = self._calc_distances2(self.Available_Rapidity,
                                                   self.Available_Phi)
        self._DeltaR2 = self.DeltaR**2
        np.fill_diagonal(angular_distances2, self._DeltaR2)
        # the distances2 array should be large enough to
        # evenually take all nodes
        self._distances2 = np.empty((self._ints.shape[0], self._ints.shape[0]),
                                    dtype=float)
        self._distances2[self._2d_avaliable_indices] = angular_distances2*kt_factor2
        
    def stopping_condition(self, idx1, idx2):
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
        return len(self._avaliable_idxs) < 2

    def update_after_join(self, idx1, idx2, idx_parent):
        """Peform updates to internal data, after combining two particles.

        Parameters
        ----------
        idx1 : int
            index of first input particle
        idx2 : int
            index of second input particle
        idx_parent : int
            index of the new particle created
        """
        new_rapidity = self.Rapidity[[idx_parent]]
        new_phi = self.Phi[[idx_parent]]
        new_angular_distance2 = self._calc_distances2(
            self.Available_Rapidity, self.Available_Phi,
            new_rapidity, new_phi)
        masked_idx_parent = self._avaliable_idxs.index(idx_parent)
        new_angular_distance2[masked_idx_parent] = self._DeltaR2
        new_pt = self.PT[[idx_parent]]
        new_kt_factor2 = self._calc_input_pt(
            self.ExpofPTInput, self.Available_PT, new_pt)**2
        new_distance2 = new_angular_distance2*new_kt_factor2
        self._distances2[idx_parent, self._avaliable_mask] = new_distance2
        self._distances2.T[idx_parent, self._avaliable_mask] = new_distance2


class Spectral(Agglomerative):
    default_params = {
                      'MaxMeanDist': 1.26,
                      'MaxMinDist': None,
                      'ExpofPTInput': 0.,
                      'ExpofPTAffinity': 0.,
                      'ExpofPTEmbedding': 0.,
                      'ExpofPTFormatInput': None,
                      'ExpofPTFormatAffinity': None,
                      'ExpofPTFormatEmbedding': None,
                      'PhyDistance': 'angular',
                      'EmbedDistance': 'root_angular',
                      'EmbedHardness': None,
                      'Laplacian': 'symmetric_carried',
                      'ExpofPhysDistance': 2.,
                      'EigenvalueLimit': 0.4,
                      'Sigma': 0.15,
                      'CutoffKNN': 5,
                      'Beta': 1.4,
                      'SingularitySuppression': 0.0001,
                      'ClipBeta': 1e-3,
                      }
    permited_values = {'MaxMeanDist': [None, Constants.numeric_classes['pdn']],
                       'MaxMinDist': [None, Constants.numeric_classes['pdn']],
                       'ExpofPTInput': Constants.numeric_classes['rn'],
                       'ExpofPTAffinity': Constants.numeric_classes['rn'],
                       'ExpofPTEmbedding': Constants.numeric_classes['rn'],
                       'ExpofPTFormatInput': [None, 'genkt', 'ratiokt',
                                              'suppressedkt'],
                       'ExpofPTFormatAffinity': [None, 'genkt', 'ratiokt',
                                                 'suppressedkt'],
                       'ExpofPTFormatEmbedding': [None, 'genkt', 'ratiokt',
                                                  'suppressedkt'],
                       'PhyDistance': ['angular'],
                       'ExpofPhysDistance': Constants.numeric_classes['pdn'],
                       'EmbedHardness': ['pt', None],
                       'EmbedDistance': ['euclidiean', 'angular', 'root_angular'],
                       'Laplacian': ['unnormed', 'symmetric', 'pt',
                                     'symmetric_carried',
                                     ],
                       'SingularitySuppression': [None, Constants.numeric_classes['pdn']],
                       'Sigma': Constants.numeric_classes['pdn'],
                       'EigenvalueLimit': Constants.numeric_classes['pdn'],
                       'CutoffKNN': [None, Constants.numeric_classes['nn']],
                       'Beta': Constants.numeric_classes['rn'],
                       'ClipBeta': Constants.numeric_classes['pdn'],
                       }
    debug_attributes = ["avaliable_mask", "laplacian", "eigenvalues",
                        "embedding", "balenced_embedding"]

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.PhyDistance == 'angular':
            self._calc_phys_distances2 = ca_distances2
        else:
            raise NotImplementedError

        if self.ExpofPTFormatInput is None:
            self._calc_input_pt = return_one
        elif self.ExpofPTFormatInput == 'genkt':
            self._calc_input_pt = genkt_factor
        elif self.ExpofPTFormatInput == 'ratiokt':
            self._calc_input_pt = ratiokt_factor
        elif self.ExpofPTFormatInput == 'suppressedkt':
            self._calc_input_pt = suppressedkt_factor
        else:
            raise NotImplementedError
        
        if self.ExpofPTFormatAffinity is None:
            self._calc_affinity_pt = return_one
        elif self.ExpofPTFormatAffinity == 'genkt':
            self._calc_affinity_pt = genkt_factor
        elif self.ExpofPTFormatAffinity == 'ratiokt':
            self._calc_affinity_pt = ratiokt_factor
        elif self.ExpofPTFormatAffinity == 'suppressedkt':
            self._calc_affinity_pt = suppressedkt_factor
        else:
            raise NotImplementedError

        if self.ExpofPTFormatEmbedding is None:
            self._calc_embedding_pt = return_one
        elif self.ExpofPTFormatEmbedding == 'genkt':
            self._calc_embedding_pt = genkt_factor
        elif self.ExpofPTFormatEmbedding == 'ratiokt':
            self._calc_embedding_pt = ratiokt_factor
        elif self.ExpofPTFormatEmbedding == 'suppressedkt':
            self._calc_embedding_pt = suppressedkt_factor
        else:
            raise NotImplementedError

        if self.Laplacian == 'pt':
            self._calc_laplacian = normalised_laplacian
            self._laplacian_norm = "PT"
        elif self.Laplacian == 'unnormed':
            self._calc_laplacian = unnormed_laplacian
            self._laplacian_norm = "Size"
        elif self.Laplacian == 'symmetric':
            self._calc_laplacian = symmetric_laplacian
            self._laplacian_norm = "Size"
        elif self.Laplacian == 'symmetric_carried':
            self._calc_laplacian = normalised_laplacian
            self._laplacian_norm = "Size"
        else:
            raise NotImplementedError
        
        if self.EmbedHardness is None:
            self._calc_hardness = return_one
        elif self.EmbedHardness == 'pt':
            self._calc_hardness = dimension_hardness
        else:
            raise NotImplementedError

        if self.Beta == 0:
            self._calc_scale = return_one
        else:
            self._calc_scale = dimension_scaling

        if self.EmbedDistance == 'euclidiean':
            self._calc_emb_distance2 = embedding_distance2
        elif self.EmbedDistance == 'angular':
            self._calc_emb_distance2 = embedding_angular2
        elif self.EmbedDistance == 'root_angular':
            self._calc_emb_distance2 = embedding_root_angular2
        else:
            raise NotImplementedError

        if self.MaxMeanDist is None:
            self._check_mean_distance = return_one
        else:
            self._check_mean_distance = mean_distance

        if self.MaxMinDist is None:
            self._check_min_distance = return_one
        else:
            self._check_min_distance = min_distance

        if self.SingularitySuppression is None:
            self._calc_singular = return_one
        else:
            self._calc_singular = singularity_factor

    def setup_internal(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Also create all debug attributes."""
        # in here we will create various permanant matrices
        self.setup_internal_local()
        # only create this space once
        space_size = (self._ints.shape[0], self._ints.shape[0])
        self._raw_distances2 = np.empty(space_size, dtype=float)
        self._distances2 = np.empty(space_size, dtype=float)
        # this will be called many times,
        # it will mostly only create small local matrices
        # but will also fill in _distances2
        self.setup_internal_global()

    def setup_internal_local(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Also create all debug attributes.
        Only includes local object, where a particle combining
        only updates a small subset of entries, rather than the whole table.
        """
        space_size = (self._ints.shape[0], self._ints.shape[0])
        # distance in physical space
        angular_distances2 = self._calc_phys_distances2(
            self.Available_Rapidity, self.Available_Phi)
        np.fill_diagonal(angular_distances2, 0.)
        #input_kt_factor2 = ratiokt_factor(
        #    self.ExpofPTInput, self.Available_PT)**2
        input_kt_factor2 = self._calc_input_pt(
            self.ExpofPTInput, self.Available_PT)**2
        physical_distance2 = angular_distances2*input_kt_factor2
        self._physical_distance2 = np.empty(space_size, dtype=float)
        self._physical_distance2[self._2d_avaliable_indices] = \
            physical_distance2

        # singularity factor
        self._singularity = np.empty(space_size, dtype=float)
        singularity = self._calc_singular(
            self.SingularitySuppression, angular_distances2, self.Available_PT)
        self._singularity[self._2d_avaliable_indices] = singularity
        # affinity
        self._affinity = np.empty((self._ints.shape[0], self._ints.shape[0]),
                                  dtype=float)
        affinity = exp_affinity(physical_distance2, sigma=self.Sigma,
                                exponent=self.ExpofPhysDistance)
        affinity *= self._calc_affinity_pt(
            self.ExpofPTAffinity, self.Available_PT)**2
        affinity *= singularity
        self._affinity[self._2d_avaliable_indices] = affinity
        affinity = self._affinity[self._2d_avaliable_indices]
        # the CutoffKNN-1 is becuase the distance to self would be 0
        knn_mask = ([] if self.CutoffKNN is None
                    else ~knn(-affinity, self.CutoffKNN-1))
        affinity[knn_mask] = 0.
        # symmetric size, always symmetric in the first step
        self._floats[self._avaliable_idxs, self._col_num["Size"]] =\
            np.sum(affinity, axis=1)

    def _embedding_distance2(self, balenced, singularity=None, pt=None):
        if pt is None:
            pt = self.Available_PT
        kt_factor2 = self._calc_embedding_pt(
            self.ExpofPTEmbedding, pt)**2
        # min(pt_i^2, pt_j^2)
        raw_distances2 = self._calc_emb_distance2(balenced) * kt_factor2
        np.fill_diagonal(raw_distances2, np.inf)
        if singularity is None:
            singularity = self._singularity[self._2d_avaliable_indices]
        distances2 = singularity*raw_distances2
        return raw_distances2, distances2

    def setup_internal_global(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Also create all debug attributes.
        Only includes global calculation where particles combining
        updates all entries, rather than a subset
        As such, this is called after each combination.
        """
        # laplacian
        affinity = self._affinity[self._2d_avaliable_indices]
        # the CutoffKNN-1 is becuase the distance to self would be 0
        knn_mask = ([] if self.CutoffKNN is None
                    else ~knn(-affinity, self.CutoffKNN-1))
        affinity[knn_mask] = 0.
        laplacian_norm = getattr(self, "Available_" + self._laplacian_norm)
        laplacian = self._calc_laplacian(affinity, laplacian_norm)
        self._laplacian = laplacian

        try:
            # embedding space
            eigenvalues, embedding = embedding_space(
                laplacian, max_eigval=self.EigenvalueLimit)
        except ValueError as e:
            elements = len(laplacian)
            if elements < 2:
                # cannot calculate eigenvectors or values
                # this is a rare case, hence the try except
                eigenvalues = np.zeros(elements)
                embedding = np.zeros((elements, elements))
            else:
                raise e
        self._eigenvalues, self._embedding = eigenvalues, embedding

        hardness = self._calc_hardness(embedding, self.Available_PT)
        scale = self._calc_scale(eigenvalues, self.Beta, self.ClipBeta)
        # problem could be embedding_norm TODO
        #balenced = scale * hardness * embedding_norm(embedding)
        balenced = scale * hardness * embedding
        self._balenced_embedding = np.empty((self._ints.shape[0],
                                             balenced.shape[1]),
                                            dtype=float)
        self._balenced_embedding[self._avaliable_idxs] = balenced

        # distance in embedding space
        self._raw_distances2[self._2d_avaliable_indices], \
            self._distances2[self._2d_avaliable_indices] = \
            self._embedding_distance2(balenced)

    def stopping_condition(self, idx1, idx2):
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
        if np.sum(self._avaliable_mask) < 2:
            return True
        distances2 = self._raw_distances2[self._2d_avaliable_indices]
        min_result = self._check_min_distance(distances2, self.MaxMinDist)
        mean_result = self._check_mean_distance(distances2, self.MaxMeanDist)
        return not (min_result * mean_result)

    def update_after_join(self, idx1, idx2, idx_parent):
        """Peform updates to internal data, after combining two particles.

        Parameters
        ----------
        idx1 : int
            index of first input particle
        idx2 : int
            index of second input particle
        idx_parent : int
            index of the new particle created
        """
        # physical distances
        new_rapidity = self.Rapidity[[idx_parent]]
        new_phi = self.Phi[[idx_parent]]
        new_angular_distance2 = self._calc_phys_distances2(
            self.Available_Rapidity, self.Available_Phi,
            new_rapidity, new_phi)
        masked_idx_parent = self._avaliable_idxs.index(idx_parent)
        new_angular_distance2[masked_idx_parent] = 0.
        new_pt = self.PT[[idx_parent]]
        new_input_kt_factor2 = self._calc_input_pt(
            self.ExpofPTInput, self.Available_PT, new_pt)**2
        new_physical_distance2 = new_angular_distance2*new_input_kt_factor2
        self._physical_distance2[idx_parent, self._avaliable_mask] = \
            new_physical_distance2
        self._physical_distance2.T[idx_parent, self._avaliable_mask] = \
            new_physical_distance2

        # new singularity
        new_singularity = self._calc_singular(
            self.SingularitySuppression, new_angular_distance2, self.Available_PT,
            new_pt, masked_idx_parent)
        self._singularity[idx_parent, self._avaliable_mask] = new_singularity
        self._singularity.T[idx_parent, self._avaliable_mask] = new_singularity

        # new affinity
        new_affinity = exp_affinity(new_physical_distance2, sigma=self.Sigma,
                                    exponent=self.ExpofPhysDistance,
                                    fill_diagonal=False)
        new_affinity[masked_idx_parent] = 0.
        new_affinity_kt_factor = self._calc_affinity_pt(
            self.ExpofPTAffinity, self.Available_PT, new_pt)**2
        new_affinity *= new_affinity_kt_factor
        self._affinity[idx_parent, self._avaliable_mask] = new_affinity
        self._affinity.T[idx_parent, self._avaliable_mask] = new_affinity

        # new size
        join_singularity = self._singularity[idx1, idx2]
        # behavior when the join is not singular
        self.Size[self._avaliable_mask] *= join_singularity
        # behavior when the join is singular
        summed_affinities = \
            np.sum(self._affinity[self._2d_avaliable_indices], axis=0)
        self.Size[self._avaliable_mask] += \
            (1 - join_singularity)*summed_affinities

        # everything else needs global calculation
        self.setup_internal_global()

    def plt_phys_distances(self, step, ax=None):
        mask_2d = self.get_historic_2d_mask(step)
        ax = self.setup_matrix_plt(ax)
        if ax is None:
            ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        distances = np.sqrt(self._physical_distance2[mask_2d])
        image = ax.imshow(distances)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Physical Distance")

    def plt_distances(self, step, ax=None):
        """As the distances will be modified each step this needs
        recalculating"""
        balenced = self.debug_data["balenced_embedding"][step]
        mask = self.debug_data["avaliable_mask"][step]
        mask_2d = self.get_historic_2d_mask(step)
        singularity = self._singularity[mask_2d]
        raw_distances2, distances2 = \
            self._embedding_distance2(balenced[mask],
                                      singularity, self.PT[mask])
        ax = self.setup_matrix_plt(ax)
        distances = np.sqrt(distances2)
        image = ax.imshow(distances)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Distance")

    def plt_affinity(self, step, ax=None):
        mask_2d = self.get_historic_2d_mask(step)
        ax = self.setup_matrix_plt(ax)
        affinity = np.sqrt(self._affinity[mask_2d])
        image = ax.imshow(affinity)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Affinity")

    def plt_laplacian(self, step, ax=None):
        ax = self.setup_matrix_plt(ax)
        laplacian = self.debug_data["laplacian"][step]
        image = ax.imshow(laplacian)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Laplacian")

    def plt_eigenvalues(self, step, ax=None):
        eigenvalues = self.debug_data["eigenvalues"][step]
        if ax is None:
            ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.scatter(range(len(eigenvalues)), eigenvalues,
                   color='k')
        ax.set_xlabel("Eigen equation")
        ax.set_ylabel("Eigenvalue")

    def plt_embedding(self, step, ax=None):
        embedding = self.debug_data["embedding"][step]
        ax = self.setup_matrix_plt(ax)
        image = ax.imshow(embedding, origin='upper')
        plt.colorbar(image, ax=ax)
        ax.set_xlabel("Eigen equation")
        ax.set_ylabel("Eigenvector element")

    def plt_balenced(self, step, ax=None):
        mask = self.debug_data["avaliable_mask"][step]
        balenced = self.debug_data["balenced_embedding"][step][mask]
        ax = self.setup_matrix_plt(ax)
        image = ax.imshow(balenced, origin='upper')
        plt.colorbar(image, ax=ax)
        ax.set_xlabel("Eigen equation")
        ax.set_ylabel("Balenced eigenvector element")

    def plt_embedding_space(self, step, dim1=0, dim2=1, ax=None):
        embedding = self.debug_data["embedding"][step]
        ax.axis('equal')
        embedding_size = np.max(np.abs(embedding[:, [dim1, dim2]]))*1.5
        mask = self.debug_data["avaliable_mask"][step]
        colours = self.make_leaf_colours()
        if ax is None:
            ax = plt.gca()
        ax.scatter([0], [0], c='k', marker='x')
        for emb, idx in zip(embedding, np.where(mask)[0]):
            descendant_idxs = self.get_decendants(last_only=True,
                                                  start_idx=idx)
            colour = colours[descendant_idxs[0]]
            ax.scatter([emb[dim1]], [emb[dim2]], color=colour)
        ax.set_xlabel(f"Eigenvector {dim1}")
        ax.set_ylabel(f"Eigenvector {dim2}")
        ax.set_xlim(-embedding_size, embedding_size)
        ax.set_ylim(-embedding_size, embedding_size)

    def plt_balenced_space(self, step, dim1=0, dim2=1, ax=None):
        mask = self.debug_data["avaliable_mask"][step]
        embedding = self.debug_data["balenced_embedding"][step][mask]
        colours = self.make_leaf_colours()
        if ax is None:
            ax = plt.gca()
        ax.scatter([0], [0], c='k', marker='x')
        for emb, idx in zip(embedding, np.where(mask)[0]):
            descendant_idxs = self.get_decendants(last_only=True,
                                                  start_idx=idx)
            colour = colours[descendant_idxs[0]]
            ax.scatter([emb[dim1]], [emb[dim2]], color=colour)
        ax.set_xlabel(f"Embedding {dim1}")
        ax.set_ylabel(f"Embedding {dim2}")

    def plt_dashboard(self, step):
        fig, ax_arr = plt.subplots(2, 4, figsize=[12, 6])
        mask = self.debug_data["avaliable_mask"][step]
        fig.suptitle(f"Step {step}, {np.sum(mask)} jets remain")
        self.plt_phys_space(step, ax=ax_arr[0, 0])
        self.plt_phys_distances(step, ax=ax_arr[1, 0])
        self.plt_affinity(step, ax=ax_arr[0, 1])
        self.plt_laplacian(step, ax=ax_arr[1, 1])
        self.plt_eigenvalues(step, ax=ax_arr[0, 2])
        self.plt_embedding(step, ax=ax_arr[1, 2])
        self.plt_embedding_space(step, ax=ax_arr[0, 3])
        self.plt_distances(step, ax=ax_arr[1, 3])
        fig.set_tight_layout(True)


class SpectralMean(Agglomerative):
    def update_after_join(self, idx1, idx2, idx_parent):
        """Peform updates to internal data, after combining two particles.

        Parameters
        ----------
        idx1 : int
            index of first input particle
        idx2 : int
            index of second input particle
        idx_parent : int
            index of the new particle created
        """
        new_balenced = 0.5*(self._balenced_embedding[idx1] +
                            self._balenced_embedding[idx2])
        self._balenced_embedding[idx_parent] = new_balenced

        # distance in embedding space
        new_distances2 = self._calc_emb_distance2(
                self._balenced_embedding[self._avaliable_idxs],
                new_balenced[np.newaxis])
        self._distances2[idx_parent, :] = new_distances2
        self._distances2[:, idx_parent] = new_distances2


class Partitional(Clustering):
    def create_int_float_tables(self, start_ints, start_floats):
        """ Format the data for clustering, allocating memory.
        The tables have space for a center point for each pottential cluster.

        Parameters
        ----------
        start_ints : list of list of int
            initial integer input data for clustering
        start_floats : list of list of floats
            initial float input data for clustering

        Returns
        -------
        ints : list of list of int
            integer input data for clustering
        floats : list of list of floats
            float input data for clustering
        """
        start_labels = [row[self._col_num["Label"]] for row in start_ints]
        assert -1 not in start_labels, "-1 is a reserved label"
        n_inputs = len(start_ints)
        # don't assume the form of start_ints
        n_unclustered = np.sum([row[self._col_num["Parent"]] == -1
                                for row in start_ints], dtype=int)
        # at worst, each point gets it's own cluster
        max_elements = n_inputs + n_unclustered
        assert max_elements <= self.memory_cap, \
            f"More particles ({n_inputs}) than possible " +\
            f"with this memory_cap ({self.memory_cap})"
        # we limit the maximum elements in memory
        ints = -np.ones((max_elements, len(self.int_columns)),
                        dtype=int)
        ints[:n_inputs] = start_ints
        floats = np.full((max_elements, len(self.float_columns)),
                         np.nan, dtype=float)
        floats[:n_inputs] = start_floats
        return ints, floats

    def run(self):
        """Perform the clustering, without storing debug_data."""
        self.setup_internal()
        jets = self.allocate()
        for jet_labels in jets:
            self.create_jet(jet_labels)

    def allocate(self):
        raise NotImplementedError

    def create_jet(self, jet_labels):
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
        # find the inputs
        sorter = np.argsort(self.Label)
        jet_idxs = sorter[np.searchsorted(self.Label, jet_labels,
                                          sorter=sorter)]
        # chose the new
        new_label = np.max(self.Label) + 1
        new_idx = self._next_free_row()
        self.Label[new_idx] = new_label
        # center objects are defined by being their own children
        # and having Parent=-1
        self.Child1[new_idx] = new_label
        self.Child2[new_idx] = new_label
        self.Rank[new_idx] = 0
        # set the parents of the jet contents
        self.Parent[jet_idxs] = new_label
        # PT px py pz eta phi energy join_distance
        self.Energy[new_idx] = np.sum(self.Energy[jet_idxs])
        self.Px[new_idx] = np.sum(self.Px[jet_idxs])
        self.Py[new_idx] = np.sum(self.Py[jet_idxs])
        self.Pz[new_idx] = np.sum(self.Pz[jet_idxs])
        self.Size[new_idx] = np.sum(self.Size[jet_idxs])
        # it's easier conceptually to calculate pt, phi and rapidity
        # afresh than derive them
        # for some reason this must be unpacked then assigned
        phi, pt = Components.pxpy_to_phipt(self.Px[new_idx], self.Py[new_idx])
        self.Phi[new_idx] = phi
        self.PT[new_idx] = pt
        self.Rapidity[new_idx] = \
            Components.ptpze_to_rapidity(self.PT[new_idx], self.Pz[new_idx],
                                         self.Energy[new_idx])

    def split(self):
        """
        Split this jet into as many unconnected jets as it contains

        Returns
        -------
        jet_list : list of Clustering
            the indervidual jets found in here
        """
        if self._ints.shape[0] == 0:  # nothing else to do if the jet is empty
            return []
        roots = np.where((self.Label != -1)*(self.Parent == -1))[0]
        jet_list = []
        jet_params = {name: getattr(self, name, self.default_params[name])
                      for name in self.default_params}
        for root in roots:
            group = np.where(self.Parent == self.Label[root])[0].tolist()
            group += [root]
            ints = self._ints[group]
            floats = self._floats[group]
            # setting the momory cap to the length of the ints
            # prevents any additional space, beyond what is
            # strictly needed being allocated to this jet
            jet = type(self)(input_data=(ints, floats),
                             jet_name=self.jet_name,
                             dict_jet_params=jet_params,
                             memory_cap=len(ints)+1)
            jet_list.append(jet)
        return jet_list


class ManualPartitional(Partitional):
    def _setup_clustering_functions(self):
        """ There is no automatic clustering """
        pass

    def run(self):
        """ There is no automatic clustering """
        raise NotImplementedError


class IterativeCone(Partitional):
    default_params = {'DeltaR': .8, 'SeedThreshold': 1.,
                      'ExpofPTFormatInput': 'min',
                      'ExpofPTInput': 0.,
                      'PhyDistance': 'angular'}
    permited_values = {'DeltaR': Constants.numeric_classes['pdn'],
                       'ExpofPTFormatInput': ['min', 'Luclus'],
                       'ExpofPTInput': Constants.numeric_classes['rn'],
                       'PhyDistance': ['angular', 'normed', 'invarient',
                                       'taxicab'],
                       'SeedThreshold': Constants.numeric_classes['pdn']}

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.PhyDistance == 'angular':
            self._calc_phys_distances2 = ca_distances2
        else:
            raise NotImplementedError

        if self.ExpofPTFormatInput is None:
            self._calc_input_pt = return_one
        elif self.ExpofPTFormatInput == 'genkt':
            self._calc_input_pt = genkt_factor
        elif self.ExpofPTFormatInput == 'ratiokt':
            self._calc_input_pt = ratiokt_factor
        else:
            raise NotImplementedError

    def setup_internal(self):
        self.deltaR2 = self.DeltaR**2

    def _select_seed(self):
        """Pick a seed particle """
        max_avaliable_idx = np.argmax(self.Available_PT)
        if self.Available_PT[max_avaliable_idx] > self.SeedThreshold:
            return max_avaliable_idx
        else:
            return -1

    def _find_jet(self):
        """ Take a single step to join pseudojets """
        # find the seed
        next_seed = self._select_seed()
        if next_seed == -1:
            # we are done
            # everything else is considered bg
            background_labels = self.Available_Label
            self._remove_background(background_labels)
            return
        # otherwise start a cone iteration
        shift = 1.
        cone_phi = self.Available_Phi[next_seed]
        cone_rapidity = self.Available_Rapidity[next_seed]
        cone_pt = self.Available_PT[next_seed]
        # this will have to be calculated at least twice anyway
        cone_energy = np.inf
        while shift > 0.01:
            cone_avaliable_idxs = \
                self._get_cone_content(cone_phi, cone_rapidity, cone_pt)
            new_cone_energy, cone_phi, cone_rapidity, cone_pt = \
                self._get_cone_kinematics(cone_avaliable_idxs)
            shift = 2*(new_cone_energy - cone_energy) \
                / (cone_energy + new_cone_energy)
            cone_energy = new_cone_energy
        cone_labels = self.Available_Label[cone_avaliable_idxs]
        return cone_labels, cone_avaliable_idxs

    def _get_cone_content(self, cone_phi, cone_rapidity, cone_pt):
        # N.B self.Phi would not work as it only gives end points
        # and lacks order gaentees
        distances2 = self._calc_phys_distances2(
            self.Available_Rapidity, self.Available_Phi,
            np.array([[cone_rapidity]]), np.array([[cone_phi]]))
        pt_factor2 = self._calc_input_pt(self.ExpofPTInput,
                                         self.Available_PT,
                                         np.array([[cone_pt]]))
        distances2 *= pt_factor2
        cone_avaliable_idxs = \
            np.where(ak.flatten(distances2) < self.deltaR2)[0]
        return cone_avaliable_idxs

    def _get_cone_kinematics(self, cone_avaliable_idxs):
        if len(cone_avaliable_idxs) == 0:
            return 0., 0., 0., 0.
        column_nums = [self._col_num[name] for name in
                       ["Energy", "Px", "Py", "Pz"]]
        columns = self._floats[self._avaliable_idxs][:, column_nums]
        e, px, py, pz = np.sum(columns, axis=0)
        phi, pt = Components.pxpy_to_phipt(px, py)
        rapidity = Components.ptpze_to_rapidity(pt, pz, e)
        return e, phi, rapidity, pt

    def allocate(self):
        jet_list = []
        while self._avaliable_idxs:
            cone_labels, cone_avaliable_idxs = self._find_jet()
            jet_list.append(cone_labels)
            self._update_avalible(self._avaliable_idxs[cone_avaliable_idxs])
        return jet_list


class SGWT(Partitional):
    default_params = {'Sigma': .11,
                      'Cutoff': 0,
                      'Normalised': True,
                      'NRounds': 15}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'Normalised': [True, False],
                       'NRounds': Constants.numeric_classes['nn']}

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        pass

    def setup_internal(self):
        """ Runs before allocate """
        self.laplacien, self.l_max_val = sgwt_functions.make_L(
                self.Leaf_Rapidity, self.Leaf_Phi,
                normalised=self.Normalised, s=self.Sigma)


    def allocate(self):
        """Sort the labels into exclusive jets"""

        available_mask = np.full_like(self.Leaf_Label, True, dtype=bool)
        jet_list = []

        # Cannot have more jets than input particles.
        max_jets = min(self.NRounds, available_mask.shape[0])

        l_idx = sgwt_functions.make_L_idx(self.Leaf_Rapidity, self.Leaf_Phi, self.Leaf_PT)

        # Precompute L_idx sum and its sorted indices
        seed_ordering = l_idx.sum(axis=0).argsort()

        unclustered_idx_pointer = 0  # Pointer to current unclustered index in seed_ordering

        round_counter = 0

        while unclustered_idx_pointer < len(seed_ordering) and round_counter < max_jets:

            if not np.any(available_mask):
                break  # will also catch the case of empty input
            
            next_unclustered_idx = seed_ordering[unclustered_idx_pointer]
            wavelet_mask = np.zeros_like(self.Leaf_Label, dtype=int)
            wavelet_mask[next_unclustered_idx] = 1

            # If the current seed is not available (already clustered), skip to the next one
            if not available_mask[next_unclustered_idx]:
                unclustered_idx_pointer += 1
                continue

            _, wp_all = sgwt_functions.wavelet_approx(self.laplacien, self.l_max_val, wavelet_mask)
            wp_all = np.array(wp_all[0].reshape(-1, 1)).flatten()
            x = sgwt_functions.min_max_scale(wp_all)

            mask = available_mask * (x < self.Cutoff)

            if np.any(mask):  # Sometimes nothing makes the cutoff.
                jet_labels = self.Leaf_Label[mask]
                jet_list.append(jet_labels)
                available_mask[mask] = False

            unclustered_idx_pointer += 1  # Move to the next unclustered particle
            round_counter += 1

        # Store anything else as a 1-particle jet
        jet_list += [self.Leaf_Label[i:i+1] for i in np.where(available_mask)[0]]

        return jet_list





def read_jet_contents(eventWise, jet_name):
    eventWise.selected_event = None
    contents = {}
    lengths = set()
    for name in Clustering.int_columns + Clustering.float_columns:
        col_name = f"{jet_name}_{name}"
        values = ak.to_list(getattr(eventWise, col_name, []))
        lengths.add(len(values))
        contents[col_name] = values
    assert len(lengths) == 1, \
        f"Some attributes of {jet_name} have differing length"
    return contents


def create_jet_contents(jet_list, existing_contents):
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
    if len(jet_list) == 0:
        return existing_contents
    try:
        jet = next(jet for jet in jet_list if jet is not None)
    except StopIteration:
        return existing_contents
    jet_name = jet.jet_name
    int_names = [f"{jet_name}_{name}" for name in jet.int_columns]
    float_names = [f"{jet_name}_{name}" for name in jet.float_columns]
    contents = {name: [[] for _ in jet_list] for name in
                int_names + float_names}
    for event_n, jets in enumerate(jet_list):
        if jets is None:
            continue
        split_jets = jets.split()
        for jet in split_jets:
            valid_rows = jet.Label != -1
            for name, values in zip(int_names, jet._ints.T):
                contents[name][event_n].append(values[valid_rows])
            for name, values in zip(float_names, jet._floats.T):
                contents[name][event_n].append(values[valid_rows])
    if existing_contents:
        contents = {name: existing_contents[name] + contents[name]
                    for name in existing_contents}
    return contents


def get_jet_params(eventWise, jet_name, add_defaults=False):
    """
    Given an eventwise in which a jet was written return it's settings.

    Parameters
    ----------
    eventWise : EventWise
        data structure with jets
    jet_name : string
        Prefix name of jet we are interested in
    add_defaults : bool
        should class default be put inplace on unspecified settings?
        (Default value = False)

    Returns
    -------
    columns : dict
        dictionary with keys being parameter names
        and values being parameter values

    """
    possible_params = set(sum([list(cluster_class.permited_values.keys())
                               for cluster_class in cluster_classes.values()],
                              []))
    prefix = jet_name
    trim = len(prefix) + 1
    columns = {name[trim:]: getattr(eventWise, name)
               for name in eventWise.hyperparameter_columns
               if name.startswith(prefix) and
               name.split('_', 1)[1] in possible_params}

    if add_defaults:
        defaults = None
        for name in cluster_classes:
            if jet_name.startswith(name):
                defaults = cluster_classes[name].default_params
                break
        if defaults is not None:
            not_found = {name: defaults[name] for name in defaults
                         if name not in columns}
            columns = {**columns, **not_found}
    return columns


def get_jet_names(eventWise):
    """
    Given and eventwise get the Prefix names of all jets inside it.

    Parameters
    ----------
    eventWise : EventWise
        data structure with jets

    Returns
    -------
    possibles : list of strings
        names of jets inside this eventWise

    """
    # grab an ending from the pseudojet int columns
    expected_ending = '_' + Agglomerative.int_columns[0]
    possibles = [name.split('_', 1)[0] for name in eventWise.columns
                 if name.endswith(expected_ending)]
    # with the word Jet put on the end
    possibles = [name for name in possibles if name.endswith("Jet")]
    return possibles


def check_hyperparameters(cluster_class, params):
    """
    Check the clustering parameters chosen are valid for the
    clustering class to be used.
    Raises a ValueError if there is a problem.

    Parameters
    ----------
    cluster_class : class
        clusteirng class defining requirements
    params : dict
        parameters to be checked

    """
    if isinstance(cluster_class, str):
        cluster_class = cluster_classes[cluster_class]
    permitted = cluster_class.permited_values
    # check all the given params are the in permitted
    if not set(params.keys()).issubset(permitted.keys()):
        unwanted_keys = [name for name in params.keys()
                         if name not in permitted]
        raise ValueError("Some parameters not permited for"
                         + f"{cluster_class.__name__}; {unwanted_keys}")
    error_str = f"In {cluster_class.__name__} {{}} is not a permitted " +\
            f"value for {{}}. Permitted value are {{}}"
    for name, opts in permitted.items():
        try:
            value = params[name]
        except KeyError:
            continue  # the default will be used
        try:
            if value in opts:
                continue  # no problem
        except TypeError:
            pass
        try:
            if Constants.is_numeric_class(value, opts):
                continue  # no problem
        except (ValueError, TypeError):
            pass
        if isinstance(opts, list):
            found_correct = False
            for opt in opts:
                try:
                    if Constants.is_numeric_class(value, opt):
                        found_correct = True
                        break
                except (ValueError, TypeError):
                    pass
            if found_correct:
                continue  # no problem
        # if we have yet ot hit a continue statment
        # then this option is not valid
        raise ValueError(error_str.format(value, name, opts))


def check_params(jet, eventWise, allow_write=True):
    """
    If the eventWise contains params, verify they are the same
    as in this jet.
    If no parameters are found in the eventWise add them.

    Parameters
    ----------
    jet : Agglomerative or dict
        the jet with parameters or just a dict of params
    eventWise : EventWise
        eventWise object to look for jet parameters in

    Returns
    -------
    : bool
        The parameters in the eventWise match those of this jet.

    """
    if isinstance(jet, tuple):
        jet_name, my_params = jet
    else:
        jet_name = jet.jet_name
        my_params = {name: getattr(jet, name, jet.default_params[name])
                     for name in jet.default_params}
    written_params = get_jet_params(eventWise, jet_name)
    if written_params:
        # if written params exist check they match jets params
        # returning false imediatly if not
        if set(written_params.keys()) != set(my_params.keys()):
            return False
        for name in written_params:
            try:
                same = np.allclose(written_params[name], my_params[name])
                if not same:
                    return False
            except TypeError:
                if not written_params[name] == my_params[name]:
                    # small bug in awkward string comparison
                    # equality works, inequality does not
                    return False
    elif allow_write:  # save the jets params
        new_hyper = {jet_name + '_' + name: my_params[name]
                     for name in my_params}
        eventWise.append_hyperparameters(**new_hyper)
    else:  # not there and we may not write it
        return False
    # if we get here everything went well
    return True


def filter_jets(eventWise, jet_name, min_jetpt=None, min_ntracks=None):
    eventWise.selected_event = None
    # decided how the pt will be filtered
    if min_jetpt is None:
        # by default all cuts at 30 GeV
        lead_jet_min_pt = other_jet_min_pt = Constants.min_pt
        # look in the eventWise for the light higgs
        flat_pids = ak.flatten(eventWise.MCPID)
        light_higgs_pid = 25
        if light_higgs_pid in flat_pids:  # if we are in a light higgs setup lower cuts
            higgs_idx = flat_pids.tolist().index(light_higgs_pid)
            light_higgs_mass = ak.flatten(eventWise.Generated_mass)[higgs_idx]
            if light_higgs_mass < 120.:
                lead_jet_min_pt = Constants.lowlead_min_pt
                other_jet_min_pt = Constants.lowother_min_pt
    else:
        lead_jet_min_pt = other_jet_min_pt = min_jetpt
    if min_ntracks is None:
        min_ntracks = Constants.min_ntracks
    # apply the pt filter
    jet_idxs = []
    jet_parent = getattr(eventWise, jet_name + "_Parent")
    try:
        jet_pt = getattr(eventWise, jet_name + "_PT")[jet_parent == -1]
    except ValueError:
        all_jet_pt = getattr(eventWise, jet_name + "_PT")
        jet_pt = []
        for i, j_pt in enumerate(all_jet_pt):
            jet_pt.append(j_pt[jet_parent[i] == -1])
        jet_pt = ak.from_iter(jet_pt)
    jet_child1 = getattr(eventWise, jet_name + "_Child1")
    empty = ak.from_iter([])
    for pts, child1s in zip(jet_pt, jet_child1):
        pts = ak.flatten(pts)
        lead_jet = np.argmax(pts)
        if lead_jet == None:
            jet_idxs.append(empty)
            continue
        if pts[lead_jet] > lead_jet_min_pt:
            pt_passes = np.where(pts > other_jet_min_pt)[0].tolist()
        else:
            jet_idxs.append(empty)
            continue
        long_enough = ak.from_iter((i for i, children in zip(pt_passes, child1s[pt_passes])
                                        if sum(children == -1) >= min_ntracks))
        jet_idxs.append(long_enough)
    return ak.from_iter(jet_idxs)


# track which classes in this module are cluster classes
def get_all_subclasses(cls):
    direct = {child.__name__: child for child in cls.__subclasses__()}
    further = {name: decendant for child in direct.values()
               for name, decendant in get_all_subclasses(child).items()}
    return {**direct, **further}

cluster_classes = get_all_subclasses(Clustering)
# track which things are valid inputs to multiapply
multiapply_input = cluster_classes

def get_jet_input_params():
    if hasattr(get_jet_input_params, "_results"):
        return get_jet_input_params._results
    
    stack = [SGWT]  #TODO don't hardcode this
    params = set()

    while stack:
        jet_class = stack.pop()
        params.update(jet_class.permited_values.keys())
        stack += jet_class.__subclasses__()

    get_jet_input_params._results = params
    return params



def cluster_multiapply(eventWise, cluster_algorithm, dict_jet_params={},
                       jet_name=None, batch_length=100, save_frequency=5.,
                       silent=False):
    """
    Apply a clustering algorithm to many events.

    Parameters
    ----------
    eventWise : EventWise
        data file with inputs, results are also written here
    cluster_algorithm: callable
        function or class that will create the jets
    dict_jet_params : dict
        dictionary of input parameters for clustering settings
        (Default value = {})
    jet_name : string
        Prefix name for the jet in eventWise
        (Default value = None)
    batch_length : int
        numebr of events to process
        (Default value = 100)
    save_frequency : float
        how often to save progress, in minutes
        (Default value = 5.)
    silent : bool
        should print statments indicating progrss be suppressed?
        useful for running in parallel
        (Default value = False)

    Returns
    -------
    : bool
        All events in the eventWise have been clustered

    """
    check_hyperparameters(cluster_algorithm, dict_jet_params)
    if jet_name is None:
        for name, algorithm in multiapply_input.items():
            if algorithm == cluster_algorithm:
                jet_name = name
                break
    if not jet_name.endswith("Jet"):
        jet_name += "Jet"
    additional_parameters = {}
    additional_parameters["jet_name"] = jet_name
    additional_parameters["run"] = True
    eventWise.selected_event = None
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(getattr(eventWise, jet_name+"_Energy", []))
    if start_point >= n_events:
        if not silent:
            print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Starting at {start_point/n_events:.1%}")
        print(f" Will stop at {end_point/n_events:.1%}")
    checked = False
    content = read_jet_contents(eventWise, jet_name)
    additonal_content = {}
    duration_name = jet_name + "_Duration"
    duration = ak.to_list(getattr(eventWise, duration_name, []))
    additonal_content[duration_name] = duration
    # set up to save
    last_save = time.time()
    save_frequency_seconds = save_frequency*60
    print_frequency = max(int((end_point-start_point)/100), 1)
    # begin loop
    jet_list = []
    for event_n in range(start_point, end_point):
        if not silent and event_n % print_frequency == 0:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        eventWise.selected_event = event_n
        try:
            start_time = time.time()
            jets = cluster_algorithm(eventWise,
                                     dict_jet_params=dict_jet_params,
                                     **additional_parameters)
            end_time = time.time()
        except (np.linalg.LinAlgError, ValueError) as e:  # handle multiple exceptions in a tuple
            # Handle the specific broadcasting ValueError
            if isinstance(e, ValueError) and str(e) == "could not broadcast input array from shape (0) into shape (0,5)":
                duration.append(np.nan)
                jet_list.append(None)
                if not silent:
                    print(f"ValueError (broadcasting issue) in event {event_n}")
            # Handle the LinAlgError
            elif isinstance(e, np.linalg.LinAlgError):
                duration.append(np.nan)
                jet_list.append(None)
                if not silent:
                    print(f"LinAlgError in event {event_n}")
            else:
                raise
            continue
        

        duration.append(end_time - start_time)
        jet_list.append(jets)
        if not checked:
            assert check_params(jets, eventWise), \
                f"Parameters don't match recorded parameters for {jet_name}"
            checked = True
        if (end_time - last_save) > save_frequency_seconds:
            content = create_jet_contents(jet_list, content)
            jet_list = []
            # periodically save
            eventWise.append(**content)
            last_save = time.time()
    eventWise.selected_event = None
    if jet_list:  # if we have some left over
        # save again at the end
        content = create_jet_contents(jet_list, content)
        eventWise.append(**additonal_content)
        eventWise.append(**content)
    return end_point == n_events

