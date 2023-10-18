import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import colors

import scipy
import scipy.sparse.linalg as ssl
from scipy.sparse import lil_matrix
from scipy.optimize import fminbound

from . import FormJets


# Deprecated, consider removing
def max_eigenvalue(L):
    """Upper-bound on the spectrum."""
    try:
        values = ssl.eigsh(L, k=1, which='LM', return_eigenvectors=False)
    except ssl.eigen.arpack.ArpackNoConvergence:
        # sparse version didn't converge.
        # try non sparse version.
        last_index = L.shape[0] - 1
        values = scipy.linalg.eigh(L, subset_by_index=[last_index, last_index],
                                   eigvals_only=True)
        # (such an elegantly consistent interface)
    return values[0]

    
def set_scales(l_min, l_max, N_scales):
    """Compute a set of wavelet scales adapted to spectrum bounds.

    Returns a (possibly good) set of wavelet scales given minimum nonzero and
    maximum eigenvalues of laplacian.

    Returns scales logarithmicaly spaced between minimum and maximum
    'effective' scales : i.e. scales below minumum or above maximum will yield
    the same shape wavelet (due to homogoneity of sgwt kernel : currently
    assuming sgwt kernel g given as abspline with t1=1, t2=2)

    Parameters
    ----------
    l_min: minimum non-zero eigenvalue of the laplacian.
       Note that in design of transform with  scaling function, lmin may be
       taken just as a fixed fraction of lmax,  and may not actually be the
       smallest nonzero eigenvalue
    l_max: maximum eigenvalue of the laplacian
    N_scales: Number of wavelets scales

    Returns
    -------
    s: wavelet scales
    """
    # Scales should be decreasing ... higher j should give larger s
    unscaled = np.logspace(2., 1., N_scales);
    scales = ((l_max - l_min)/90.) * (unscaled - 10.) + l_min
    return scales

def kernel(x, g_type='abspline', a=2, b=2, t1=1, t2=2):
    """Compute sgwt kernel.

    This function will evaluate the kernel at input x

    Parameters
    ----------
    x : independent variable values
    type : 'abspline' gives polynomial / spline / power law decay kernel
    a : parameters for abspline kernel, default to 2
    b : parameters for abspline kernel, default to 2
    t1 : parameters for abspline kernel, default to 1
    t2 : parameters for abspline kernel, default to 2

    Returns
    -------
    g : array of values of g(x)
    """
    if g_type == 'abspline':
        g = kernel_abspline3(x, a, b, t1, t2)
    elif g_type == 'mh':
        g = x * np.exp(-x)
    else:
        print ('unknown type')
        #TODO Raise exception

    return g


def kernel_abspline3(x, alpha, beta, t1, t2):
    """Monic polynomial / cubic spline / power law decay kernel

    Defines function g(x) with g(x) = c1*x^alpha for 0<x<x1
    g(x) = c3/x^beta for x>t2
    cubic spline for t1<x<t2,
    Satisfying g(t1)=g(t2)=1

    Parameters
    ----------
    x : array of independent variable values
    alpha : exponent for region near origin
    beta : exponent decay
    t1, t2 : determine transition region


    Returns
    -------
    r : result (same size as x)
"""
    # Convert to array if x is scalar, so we can use fminbound
    if np.isscalar(x):
        x = np.array(x, ndmin=1)

    r = np.zeros(x.size)

    # Compute spline coefficients
    # M a = v
    M = np.array([[1, t1, t1**2, t1**3],
                  [1, t2, t2**2, t2**3],
                  [0, 1, 2*t1, 3*t1**2],
                  [0, 1, 2*t2, 3*t2**2]])
    v = np.array([[1],
                  [1],
                  [t1**(-alpha) * alpha * t1**(alpha - 1)],
                  [-beta * t2**(-beta - 1) * t2**beta]])
    a = np.linalg.lstsq(M, v, rcond=None)[0]

    r1 = np.logical_and(x>=0, x<t1).nonzero()
    r2 = np.logical_and(x>=t1, x<t2).nonzero()
    r3 = (x>=t2).nonzero()
    r[r1] = x[r1]**alpha * t1**(-alpha)
    r[r3] = x[r3]**(-beta) * t2**(beta)
    x2 = x[r2]
    r[r2] = a[0]  + a[1] * x2 + a[2] * x2**2 + a[3] * x2**3

    return r

  
def filter_design(l_max, N_scales, design_type='default', lp_factor=20,
                  a=2, b=2, t1=1, t2=2):
    """Return list of scaled wavelet kernels and derivatives.
    Note: see page 26 of the paper for the choice of values here.
    
    g[0] is scaling function kernel, 
    g[1],  g[Nscales] are wavelet kernels

    Parameters
    ----------
    l_max : upper bound on spectrum
    N_scales : number of wavelet scales
    design_type: 'default' or 'mh'
    lp_factor : lmin=lmax/lpfactor will be used to determine scales, then
       scaling function kernel will be created to fill the lowpass gap. Default
       to 20.

    Returns
    -------
    g : scaling and wavelets kernel
    gp : derivatives of the kernel (not implemented / used)
    t : set of wavelet scales adapted to spectrum bounds
    """
    g = []
    gp = []
    l_min = l_max / lp_factor
    t = set_scales(l_min, l_max, N_scales)
    if design_type == 'default':
        # Find maximum of gs. Could get this analytically, but this also works
        f = lambda x: -kernel(x, a=a, b=b, t1=t1, t2=t2)
        x_star = fminbound(f, 1, 2)
        gamma_l = -f(x_star)
        g.append(lambda x: gamma_l * np.exp(-(x)))
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, a=a, b=b, t1=t1,t2=t2))
    elif design_type == 'mh':
        l_min_fac = 0.4 * l_min
        g.append(lambda x: 1.2 * np.exp(-1) * np.exp(-(x/l_min_fac)**4))
        for scale in t:
            g.append(lambda x,s=scale: kernel(s * x, g_type='mh'))
    else:
        print ('Unknown design type')
        raise NotImplementedError(f"Unknown design type {design_type}")
        
    return (g, gp, t)


def cheby_coeff(g, m, N=None, arange=(-1,1)):
    """ Compute Chebyshev coefficients of given function using numpy's chebfit.

    Parameters
    ----------
    g : function handle, should define function on arange
    m : maximum order Chebyshev coefficient to compute
    N : grid order used to compute quadrature (default is m+1)
    arange : interval of approximation (defaults to (-1,1) )

    Returns
    -------
    c : list of Chebyshev coefficients, ordered such that c(j+1) is 
      j'th Chebyshev coefficient
    """
    if N is None:
        N = m+1

    print(f"g = {g}")
    print(f"m = {m}")
    print(f"N = {N}")
    print(f"arange = {arange}")

    x = np.linspace(arange[0], arange[1], N)
    y = g(x)
    
    c = np.polynomial.chebyshev.chebfit(x, y, m)

    return c


def preprocess_coefficients(c):
    """Preprocess coefficients to ensure they're in the correct format."""
    # If c is a scalar or 1D list/array, convert to 2D array

    if np.isscalar(c):
        return np.array([[c]])

    if len(c) == 0:
        raise ValueError("Coefficients are an empty list.")
    
    # Check sizes of inner arrays of c and raise error if they differ
    if isinstance(c, list) and any(isinstance(x, np.ndarray) for x in c):
        sizes = [x.shape[0] for x in c if isinstance(x, np.ndarray)]
        if len(set(sizes)) > 1:
            raise ValueError("All inner arrays of c must be of the same size.")

    if isinstance(c, (list, np.ndarray)) and np.ndim(c) == 1:
        return np.array([c])
    
    # If c is a list of arrays, stack them
    if isinstance(c, list) and any(isinstance(x, np.ndarray) for x in c):
        if len(set(x.shape[0] for x in c)) > 1:
            raise ValueError("All inner arrays of c must be of the same size.")
        return np.vstack(c)

    return c.astype(np.float64)


def cheby_op(wavelet_delta, laplacian, chebyshef_coefficients, arange):
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev basis) applied to input."""
    # Ensure everything is float type
    wavelet_delta = wavelet_delta.astype(np.float64)
    laplacian = laplacian.astype(np.float64)

    chebyshef_coefficients = preprocess_coefficients(chebyshef_coefficients)
    
    N_scales, N_coefficients = chebyshef_coefficients.shape

    half_width = (arange[1] - arange[0]) / 2.0
    center = (arange[1] + arange[0]) / 2.0

    fourier_transform_old = wavelet_delta
    fourier_transform_cur = (laplacian.dot(wavelet_delta) - center * wavelet_delta) / half_width

    # Preallocate results
    results = [np.zeros_like(wavelet_delta) for _ in range(N_scales)]
    for j in range(N_scales):
        results[j] += 0.5 * chebyshef_coefficients[j, 0] * fourier_transform_old
        if N_coefficients > 1:
            results[j] += chebyshef_coefficients[j, 1] * fourier_transform_cur

    for k in range(2, N_coefficients):
        fourier_transform_new = (2/half_width)*(laplacian.dot(fourier_transform_cur)
                                                - center*fourier_transform_cur) \
                                - fourier_transform_old
        for j in range(N_scales):
            results[j] += chebyshef_coefficients[j, k] * fourier_transform_new

        fourier_transform_old, fourier_transform_cur = fourier_transform_cur, fourier_transform_new

    if np.all(laplacian == 0):
        warnings.warn("The Laplacian matrix laplacian is all zeros.")
        results = [np.zeros_like(wavelet_delta) for _ in
                   range(chebyshef_coefficients.shape[0])]
    
    return results



def make_L_idx(particle_rapidities, particle_phis, particle_pts):
    """ Makes mask for where to place wavelet in the particle array.

    Parameters
    ----------
    rapidity : array of float
        Row of rapidity values.
    phi : array of float
        Row of phi values.
    particle_pts : array of float 
        row of pts values.
    idx_no : int (optional)
        Index of summed anti-kt weight for each particle, 
        sorted from smallest to largest

    Returns
    -------
    L_idx : distance array determined by the anti-kt metric
    """

    ca_distances = np.sqrt(FormJets.ca_distances2(particle_rapidities, particle_phis))
    gen_kt_factor = FormJets.genkt_factor(-1, np.array(particle_pts, dtype=float))
    L_idx = ca_distances * gen_kt_factor

    return L_idx


def make_L(particle_rapidities, particle_phis, normalised=True, sigma = 0.15):
    """ Makes a weighted Laplacian from particle rapidities and phis,
    using the method found in CALE paper for swiss roll example.

    Parameters
    ----------
    rapidity : array of float
        Row of rapidity values.
    phi : array of float
        Row of phi values.
    sigma : int (optional)
        (sigma) level of weight scaling in the graph,
        larger values means points further away from
        delta will have larger coefficients 

    Returns
    -------
    L : n x n array of float

    l_max_val: float
        Largest eigenvalue
    """

    distances2 = FormJets.ca_distances2(particle_rapidities, particle_phis)
    print(f"distances2 are {distances2}")
    print(f"distances are {np.sqrt(distances2)}")
    affinities = FormJets.exp_affinity(distances2, sigma)
    print(f"affinities are {affinities}")
    #affinities = np.exp(-distances2**2 / (2 * sigma**2))
    #np.fill_diagonal(affinities, 0.)
    diagonals = np.diag(np.sum(affinities, axis=0))
    laplacian = diagonals - affinities

    if normalised:
        #laplacian = (np.linalg.inv(diagonals)**0.5) @ laplacian @ (np.linalg.inv(diagonals)**0.5)
        diags = np.sum(affinities, axis=0)
        diags_sqrt = 1.0 / np.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        diagonals = np.diag(diags_sqrt)
        laplacian = diagonals @ (laplacian @ diagonals)
        l_max_val = 2.
    else:
        l_max_val = max_eigenvalue(laplacian)

    return laplacian, l_max_val


# called without optional arguments in CALEFormJets
def wavelet_approx(L, l_max_val, L_idx,  N_scales = 1, m = 50):
    """ Approximates wavelets of N_scales around point given by L_idx

    Parameters
    ----------
    L : n x n array of float
        weighted graph Laplacian constructed by the method found in
        8.2 of CALE paper
    l_max_val : float
    N_scales : int (optional)
        Number of scales in the filter bank
    m: int (optional)
        degree to calculate the approximating polynomial 

    Returns
    -------
    d : n x n array of int
        reshaped L_idx, used for plot colouring

    wp_all: N_scales x n x n array of float 
        coefficients of shifted chebyshev polynomials
    """

    # Design filters for transform
    (g, gp, _) = filter_design(l_max_val, N_scales)
    arange = (0.0, l_max_val)

    # Chebyshev polynomial approximation
    c = [cheby_coeff(g[i], m, m+1, arange) for i in range(len(g))]
    N = L.shape[0]

    # Compute transform of delta at one vertex
    # Vertex to center wavelets to be shown
    d = L_idx.reshape(-1,1)
    print(f"chebyshef_coefficients = {c}")

    # forward transform, using chebyshev approximation
    wp_all = cheby_op(d, L, c, arange)

    return d, wp_all


def min_max_scale(x):
    x1 = 2*(x - min(x)) / ( max(x) - min(x) )-1
    return x1



def cluster_particles(particle_rapidities, particle_phis, particle_pts, s=0.11, cutoff=0, rounds=15, m=50, normalised=True):
    """Cluster particles based on their wavelet properties"""

    # Number of particles
    num_particles = len(particle_rapidities)

    # Initialize the clusters array and cluster list
    clusters = np.zeros(num_particles)
    cluster_list = []

    # Generate the laplacian matrix
    L, l_max_val = make_L(particle_rapidities, particle_phis, normalised=normalised, sigma=s)

    # Convert particle data to appropriate format
    particle_data = np.array((particle_rapidities, particle_phis, particle_pts))
    
    # Generate the L_idx matrix
    L_idx = make_L_idx(*particle_data)

    # Precompute L_idx sum and its sorted indices
    seed_ordering = L_idx.sum(axis=0).argsort()

    # Initialize pointers and counters
    unclustered_idx_pointer = 0
    round_counter = 0

    # Loop until all particles are clustered or until we reach the max number of rounds
    while unclustered_idx_pointer < len(seed_ordering) and round_counter < rounds:
        
        # Get the index for the next unclustered particle
        next_unclustered_idx = seed_ordering[unclustered_idx_pointer]
        
        # Create the wavelet mask
        wavelet_mask = np.zeros_like(particle_rapidities, dtype=int)
        wavelet_mask[next_unclustered_idx] = 1

        # If the current seed is already clustered, move on to the next seed
        if clusters[next_unclustered_idx] != 0:
            unclustered_idx_pointer += 1
            continue

        # Compute the wavelet transform
        _, wp_all = wavelet_approx(L, 2, wavelet_mask, m=m)
        wavelet_values = min_max_scale(np.array(wp_all[0])).flatten()
        
        # Find particles below the cutoff and those available for clustering
        below_cutoff_indices = set(np.where(wavelet_values < cutoff)[0])
        available_particles = set(np.where(clusters == 0)[0])
        
        # Determine labels of particles to cluster in this round
        labels = list(below_cutoff_indices & available_particles)

        # Update the clusters array and cluster list
        lbl = len(cluster_list) + 1  # Current cluster number
        clusters[labels] = lbl
        cluster_list.append(labels)

        # Increment pointers and counters
        unclustered_idx_pointer += 1
        round_counter += 1

    return clusters, cluster_list

