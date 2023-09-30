import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import colors

import scipy
import scipy.sparse.linalg as ssl
from scipy.sparse import lil_matrix
from scipy.optimize import fminbound



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


def rescale_laplacian(L, lmax):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def rough_l_max(L):
    """Return a rough upper bound on the maximum eigenvalue of L.

    Parameters
    ----------
    L: Symmetric matrix

    Return
    ------
    l_max_ub: An upper bound of the maximum eigenvalue of L.
    """
    # TODO: Check if L is sparse or not, and handle the situation accordingly

    l_max = np.linalg.eigvalsh(L.todense()).max()

    # TODO: Fix this
    # At least for demo_1, this is much slower
    #l_max = ssl.arpack.eigsh(L, k=1, return_eigenvectors=False,
    #                         tol=5e-3, ncv=10)

    l_max_ub =  1.01 * l_max
    return l_max_ub

    
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
    x1=1
    x2=2
    s_min = x1 / l_max
    s_max = x2 / l_min
    # Scales should be decreasing ... higher j should give larger s
    s = np.exp(np.linspace(np.log(s_max), np.log(s_min), N_scales));

    return s

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
    a = np.linalg.lstsq(M, v)[0]

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
    """ Compute Chebyshev coefficients of given function.

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

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N+1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m+1)
    for j in range(m+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c


def cheby_op(f, L, c, arange):
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev
    basis) applied to input.

    Coefficients for multiple polynomials may be passed as a list. This
    is equivalent to setting
    r[0] = cheby_op(f, L, c[0], arange)
    r[1] = cheby_op(f, L, c[1], arange)
    ...
 
    but is more efficient as the Chebyshev polynomials of L applied to f can be
    computed once and shared.

    Parameters
    ----------
    f : input vector
    L : graph laplacian (should be sparse)
    c : Chebyshev coefficients. If c is a plain array, then they are
       coefficients for a single polynomial. If c is a list, then it contains
       coefficients for multiple polynomials, such  that c[j](1+k) is k'th
       Chebyshev coefficient the j'th polynomial.
    arange : interval of approximation

    Returns
    -------
    r : If c is a list, r will be a list of vectors of size of f. If c is
       a plain array, r will be a vector the size of f.    
    """
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op(f, L, [c], arange)
        return r[0]

    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0

    Twf_old = f
    Twf_cur = (L*f - a2*f) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        # This is the polynomial that approximates the Laplacian
        Twf_new = (2/a1) * (L*Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                #this is shifted chebyshev polynomial coeff at scale j
                r[j] = r[j] + c[j][k+1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r, Twf_old


def framebounds(g, lmin, lmax):
    """

    Parameters
    ----------
    g : function handles computing sgwt scaling function and wavelet
       kernels
    lmin, lmax : minimum nonzero, maximum eigenvalue

    Returns
    -------
    A , B : frame bounds
    sg2 : array containing sum of g(s_i*x)^2 (for visualization)
    x : x values corresponding to sg2
    """
    N = 1000 # number of points for line search
    x = np.linspace(lmin, lmax, N)
    Nscales = len(g)

    sg2 = np.zeros(x.size)
    for ks in range(Nscales):
        sg2 += (g[ks](x))**2

    A = np.min(sg2)
    B = np.max(sg2)

    return (A, B, sg2, x)

def view_design(g, t, arange):
    """Plot the scaling and wavelet kernel.

    Plot the input scaling function and wavelet kernels, indicates the wavelet
    scales by legend, and also shows the sum of squares G and corresponding
    frame bounds for the transform.

    Parameters
    ----------
    g : list of  function handles for scaling function and wavelet kernels
    t : array of wavelet scales corresponding to wavelet kernels in g
    arange : approximation range

    Returns
    -------
    h : figure handle
    """
    x = np.linspace(arange[0], arange[1])
    h = plt.figure()
    
    J = len(g) 
    G = np.zeros(x.size)

    for n in range(J):
        if n == 0:
            lab = 'h'
            plt.plot(x, g[n](x), label=lab)
            G += g[n](x)

    #plt.plot(x, G, 'k', label='G')

    #(A, B, _, _) = framebounds(g, int(arange[0]), int(arange[1]))
    #plt.axhline(A, c='m', ls=':', label='A')
    #plt.axhline(B, c='g', ls=':', label='B')
    #plt.xlim(arange[0], arange[1])

    plt.title('Scaling function kernel h(x), Wavelet kernels g(t_j x) \n'
              'sum of Squares G, and Frame Bounds')
    #plt.yticks(np.r_[0:4])
    #plt.ylim(0, 3)
    plt.legend()

    return h


def swiss_roll(n, a=1, b=4, depth=5, do_rescale=True):
    """Return n random points laying in a swiss roll.
    The swiss roll manifold is the manifold typically used in manifold learning
    and other dimensionality reduction techniques. It is determined by the
    parametric equations
    x1 = pi * sqrt((b^2 - y^2)*t + a^2) * cos(pi * sqrt((b^2 - y^2)*t1 + a^2))
    x2 = depth * t2
    x3 = pi * sqrt((b^2 - y^2)*t + a^2) * sin(pi * sqrt((b^2 - y^2)*t1 + a^2))
    
    Parameters
    ----------
    n : Number of points
    a : Initial angle is a*pi
    b : End angle is b*pi
    depth: Depth of the roll
    do_rescale: If True, rescale to the plus/minus 1 range, default to True
    
    Returns
    -------
    x : A 3-by-n ndarray [x1; x2; x3] with the points from the roll
    """
    y = np.random.rand(2, n)
    t = np.pi * np.sqrt((b*b - a*a) * y[0,:] + a*a)
    x2 = depth * y[1,:]
    x1 = t * np.cos(t)
    x3 = t * np.sin(t)

    if do_rescale:
        x1 = rescale(x1)
        x2 = rescale(x2)
        x3 = rescale(x3)
    
    x = np.vstack([x1, x2, x3])

    return x


def rescale(x):
    """Rescale vector x into the [-1, 1] range.
    Parameters
    ----------
    x: 1 dimensional ndarray
    Returns:
    -------
    x: The original vector rescale to the range [-1, 1]
    """
    x -= x.mean()
    x /= np.max(np.abs(x))
    return x


def clean_axes(ax):
    for a in ax.w_xaxis.get_ticklines()+ax.w_xaxis.get_ticklabels():
        a.set_visible(False)
    for a in ax.w_yaxis.get_ticklines()+ax.w_yaxis.get_ticklabels():
        a.set_visible(False)
    for a in ax.w_zaxis.get_ticklines()+ax.w_zaxis.get_ticklabels():
        a.set_visible(False) 
    return


# TOD import
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
    rapidity = np.array(rapidity)
    phi = np.array(phi)

    if rapidity_column is None:
        rapidity_column = np.expand_dims(rapidity, 1)
    rapidity_distances = np.abs(rapidity - rapidity_column)
    if phi_column is None:
        phi_column = np.expand_dims(phi, 1)
    phi_distances = angular_distance(phi, phi_column)
    distances2 = phi_distances ** 2 + rapidity_distances ** 2
    return distances2


#TODO import
def angular_distance(a, b):
    """
    Get the shortest distance between a and b

    Parameters
    ----------
    a : float or arraylike of floats
        angle
        
    b : float or arraylike of floats
        angle
        

    Returns
    -------
    : float or arraylike of floats
        absolute distance between angles

    """
    raw = a - b
    return np.min((raw%(2*np.pi), np.abs(-raw%(2*np.pi))), axis=0)

# TODO import
def make_kT_mat(y, phi, pT):
    # exponent exp1 is -1 for anti-kT, 0 for deltaRmatrix (CA), 1 for kT
    nparts = len(pT)
    mat = np.zeros((nparts, nparts))
    for i in range(nparts):
        for j in range(i+1,nparts):
                multiplier = 1/np.max((pT[i], pT[j]))
                mat[i,j]=(multiplier*np.sqrt(2*(np.cosh(y[i] - y[j]) - np.cos(phi[i]-phi[j]))))
    return 0.5 * (mat + mat.T)


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

    L_idx = make_kT_mat(particle_rapidities, particle_phis, particle_pts)

    return L_idx


#TODO import
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


def make_L(particle_rapidities, particle_phis, normalised=True, s = 0.15):

    """ Makes a weighted Laplacian from particle rapidities and phis,
    using the method found in SGWT paper for swiss roll example.

    Parameters
    ----------
    rapidity : array of float
        Row of rapidity values.
    phi : array of float
        Row of phi values.
    s : int (optional)
        (sigma) level of weight scaling in the graph,
        larger values means points further away from
        delta will have larger coefficients 

    Returns
    -------
    L : n x n array of float

    l_max_val: float
        Largest eigenvalue
    """

    d = ca_distances2(particle_rapidities, particle_phis)
    A = exp_affinity(d, s)
    #A = np.exp(-d**2 / (2 * s**2))
    #np.fill_diagonal(A, 0.)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    if normalised == True:
        #L = (np.linalg.inv(D)**0.5) @ L @ (np.linalg.inv(D)**0.5)
        diags = np.sum(A, axis=0)
        diags_sqrt = 1.0 / np.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        D = np.diag(diags_sqrt)
        L = D @ (L @ D)

    l_max_val = max_eigenvalue(L)
    L = rescale_laplacian(L, l_max_val)

    return L, l_max_val



def wavelet_approx(L, l_max_val, L_idx,  N_scales = 1, m = 50):

    """ Approximates wavelets of N_scales around point given by L_idx

    Parameters
    ----------
    L : n x n array of float
        weighted graph Laplacian constructed by the method found in
        8.2 of SGWT paper
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
    (g, gp, t) = filter_design(l_max_val, N_scales)
    arange = (0.0, l_max_val)

    # Chebyshev polynomial approximation
    c = [cheby_coeff(g[i], m, m+1, arange) for i in range(len(g))]
    N = L.shape[0]

    # Compute transform of delta at one vertex
    # Vertex to center wavelets to be shown
    d = L_idx.reshape(-1,1)

    # forward transform, using chebyshev approximation
    wp_all, Twf_cur = cheby_op(d, L, c, arange)

    return d, wp_all


def plot_wavelet(particle_rapidities, particle_phis, wp_all, n=0, save_fig=False, plt_title='Wavelet', fig_name = 'wavelet_results.png'):

    """ Plot of wavelet convoluted with the graph
    Laplacian at a particular scale (n)

    Parameters
    ----------
    rapidity : array of float
        Row of rapidity values.
    phi : array of float
        Row of phi values.
    wp_all : N_scapes x n x n array of float 
        coefficients of shifted chebyshev polynomials
    n: int (optional)
        wavlet scale, 0 is scaling function

    Returns
    -------
    plot of particles coloured by their shifted chebyshev coeff 
    """

    plt.scatter(particle_rapidities, particle_phis, c=[wp_all[n]], cmap='viridis')
    plt.rcParams["figure.figsize"] = (8,8)
    plt.title(plt_title)

    if save_fig == True:
        plt.savefig(fig_name)
    plt.show()


def cluster_L_plot(particle_rapidities, particle_phis, lbls, legend=True, savfig=False):

    data = np.array((particle_rapidities, particle_phis, lbls))
    x= data[0,:]
    y= data[1,:]

    uniq = list(set(data[-1,:]))

    z = range(1,len(uniq))
    cmap = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for i in range(len(uniq)):
        indx = data[-1,:] == uniq[i]
        plt.scatter(x[indx], y[indx], s=15, color=scalarMap.to_rgba(i), label=uniq[i], alpha=0.9)

    plt.rcParams["figure.figsize"] = (9.5,9.5)
    plt.xlabel('Rapidity')
    plt.ylabel('phi')
    plt.title('Clusters')
    if legend == True:
        plt.legend(loc='upper right')

    if savfig==True:
        plt.savefig('wavelet_clusters.png')

    plt.show()


def min_max_scale(x):
    x1 = 2*(x - min(x)) / ( max(x) - min(x) )-1
    return x1


def wavelet_pos(L_idx, i):
    """ Return the i'th highest index particle, as determined by
    summing across the rows of L_idx

    Parameters
    ----------
    L_idx : array of float
        distance array of L_idx.
    i : int
        index to return.


    Returns
    -------
    1d binary array of index to place the wavelet  
    """

    idx = L_idx.sum(axis=0).argsort()[i]
    wav_pos = np.zeros_like(L_idx.sum(axis=0), dtype=int)
    wav_pos[idx] = 1

    return wav_pos


def cluster_particles(particle_rapidities, particle_phis, particle_pts, s=0.11, cutoff=0, rounds=5, m=50, normalised=True):
    """Cluster particles based on their wavelet properties"""

    # Number of particles
    num_particles = len(particle_rapidities)

    # Initialize the clusters array and cluster list
    clusters = np.zeros(num_particles)
    cluster_list = []

    # Generate the laplacian matrix
    L, l_max_val = make_L(particle_rapidities, particle_phis, normalised=normalised, s=s)

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

