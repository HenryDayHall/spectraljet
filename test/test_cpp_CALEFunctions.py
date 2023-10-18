import numpy as np
from numpy import testing as tst
from spectraljet import Constants, FormJets, CALEFunctions
from spectraljet.cpp_CALE import build

build_dir = Constants.CALE_build_dir
build.build(build_dir, force_rebuild = True)
CALE = build.get_module(build_dir)

from . import test_CALEFunctions, test_Components, test_FormJets


def test_LaplacianWavelet():
    # shouldn't choke on an empty laplacian
    found = CALE.LaplacianWavelet([], [1.], 0, [-1, 1])
    assert len(found) == 0
    # This can be compared to cheby_op, as they fill the same role
    intervals = [[-1, 1], [0, 2]]
    laplacians = ([[1, -0.1], [-0.1, 1]],
                  [[10, -0.1], [-0.1, 10]],
                  [[1, 0., -0.5], [0., 1, 0.], [-0.5, 0., 1]],
                  [[1, -0.1, -0.5, 0.],
                   [-0.1, 1., 0., -0.5],
                   [-0.5, 0., 1., -0.1],
                   [0., -0.5, -0.1, 1.]])
    coefficients = CALE.ChebyshevCoefficients(4);
    for interval in intervals:
        for laplacian in laplacians:
            num_points = len(laplacian)
            for seed in range(num_points):
                found = CALE.LaplacianWavelet(laplacian, coefficients[:num_points], seed, interval)
                wavelet_mask = np.zeros(num_points, dtype=int)
                wavelet_mask[seed] = 1
                expected = CALEFunctions.cheby_op(wavelet_mask, np.array(laplacian),
                                                  coefficients[:num_points], interval)
                error_message = f"interval = {interval}, seed = {seed}, " + \
                                f"coefficients = {coefficients[:num_points]}, " + \
                                f"laplacian = {laplacian}"
                tst.assert_allclose(found, expected[0], err_msg=error_message, atol=1e-5)



def test_ChebyshevCoefficients_vs_np_cheb():
    # There is no point importing the chebyshev tests,
    # as they are for application to a generic function,
    # but we specifically want the coefficients of f(x) = exp(-x)
    # Using numpy's chebfit function to compute the coefficients
    # For the function f(x) = x^3
    def f(x):
        return np.exp(-x)

    grid_order = 400
    interval_min = -1
    interval_max = 1
    x = np.linspace(interval_min, interval_max, grid_order)
    np_coeffs = np.polynomial.chebyshev.chebfit(x, f(x), 5)
    coefficients = CALE.ChebyshevCoefficients(5, grid_order, interval_min, interval_max)
    tst.assert_allclose(coefficients, np_coeffs, atol=1e-4)


class TestCPPMakeLIdx(test_CALEFunctions.TestMakeLIdx):
    def function(self, particle_rapidities, particle_phis, particle_pts):
        particle_rapidities = list(particle_rapidities)
        particle_phis = list(particle_phis)
        particle_pts = list(particle_pts)
        metric = CALE.JetMetrics.antikt
        distances2 = CALE.NamedDistance2Matrix(particle_pts, particle_rapidities, particle_phis, metric)
        return np.sqrt(distances2)


def test_VectorAddition():
    # Two empty vectors should make another empty vector
    assert len(CALE.VectorAddition([], [])) == 0
    # Two vectors with 0 in should make another vectors with 0 in
    tst.assert_allclose(CALE.VectorAddition([0,0,0], [0,0,0]), [0,0,0])
    # Two vectors with 1 in should make another vectors with 2 in
    tst.assert_allclose(CALE.VectorAddition([1,1,1], [1,1,1]), [2,2,2])
    # A vector of -1 and a vector of 1 should make a vector of 0
    tst.assert_allclose(CALE.VectorAddition([-1,-1,-1], [1,1,1]), [0,0,0])
    # One last test
    tst.assert_allclose(CALE.VectorAddition([1,2,3], [4,5,6]), [5,7,9])


def test_MatrixDotVector():
    # Empty matrix and empty vector should make an empty vector
    assert len(CALE.MatrixDotVector([], [])) == 0
    # Length one matrix and length one vector should make a length one vector
    tst.assert_allclose(CALE.MatrixDotVector([[1]], [1]), [1])
    # create a range of matrices and a range of vectors
    length_2_matrices = ([[0,0],[0,0]],
                         [[1,2],[3,4]],
                         [[-1,2],[-3,4]],
                         [[5,6],[7,8]],
                         [[5,-6],[-7,8]],
                         [[9,10],[11,12]])
    length_2_vectors = ([0,0],
                        [1,2],
                        [-1,-2],
                        [3,-4],
                        [5,6])
    # Test all combinations of the above against numpy
    for matrix in length_2_matrices:
        for vector in length_2_vectors:
            tst.assert_allclose(CALE.MatrixDotVector(matrix, vector), np.dot(matrix, vector))
    # again but length 3
    length_3_matrices = ([[0,0,0],[0,0,0],[0,0,0]],
                         [[1,2,3],[4,5,6],[7,8,9]])
    length_3_vectors = ([0,0,0],
                        [1,2,3])
    # Test all combinations of the above against numpy
    for matrix in length_3_matrices:
        for vector in length_3_vectors:
            tst.assert_allclose(CALE.MatrixDotVector(matrix, vector), np.dot(matrix, vector))



class TestCPPAngularDistance(test_Components.TestAngularDistance):
    def function(self, a, b):
        if hasattr(a, '__iter__'):
            return [CALE.AngularDistance(a_i, b_i) for a_i, b_i in zip(a, b)]
        return CALE.AngularDistance(a, b)


def test_CambridgeAachenDistance2():
    # Too complicated to import existing test due to matrix structures;
    # just test a few cases
    found = CALE.CambridgeAachenDistance2(0, 0, 0, 0)
    tst.assert_allclose(found, 0)
    found = CALE.CambridgeAachenDistance2(1, 0, 0, 0)
    tst.assert_allclose(found, 1)
    found = CALE.CambridgeAachenDistance2(0, 1, 0, 0)
    tst.assert_allclose(found, 1)
    found = CALE.CambridgeAachenDistance2(0, 0, 1, 0)
    tst.assert_allclose(found, 1)
    found = CALE.CambridgeAachenDistance2(0, 0, 0, 1)
    tst.assert_allclose(found, 1)
    found = CALE.CambridgeAachenDistance2(1, 1, 0, 0)
    tst.assert_allclose(found, 2)
    found = CALE.CambridgeAachenDistance2(1, 0, 1, 0)
    tst.assert_allclose(found, 0)
    found = CALE.CambridgeAachenDistance2(1, 0, 0, 1)
    tst.assert_allclose(found, 2)
    found = CALE.CambridgeAachenDistance2(2.5, 0, 0, 0)
    tst.assert_allclose(found, 6.25)
    found = CALE.CambridgeAachenDistance2(0, -2.5, 0, 0)
    tst.assert_allclose(found, 6.25)
    found = CALE.CambridgeAachenDistance2(0, np.pi, 0, 1.-np.pi)
    tst.assert_allclose(found, 1.)


def test_GeneralisedKtDistance2():
    # Its too complicated to import the existing test due to matrix structures;
    # But we can compare outputs with the python version
    input_rapidities = [0, 1., 2., 3., -1., -2., -3.]
    input_rapidities += input_rapidities
    input_phis = [0, 1., np.pi, np.pi - 1., -1.,  2*np.pi, 0.]
    input_phis += input_phis[::-1]
    input_pts = [0, 1., 2., 0., 1., 10., 100.]
    input_pts += input_pts
    n_particles = len(input_rapidities)
    ca2_from_python = FormJets.ca_distances2(input_rapidities, input_phis)
    metric_dict = {-1: CALE.JetMetrics.antikt, 0: CALE.JetMetrics.cambridge_aachen, 1: CALE.JetMetrics.kt}
    for kt_factor in [-1, 0, 0.5, 1]:
        kt_from_python = FormJets.genkt_factor(kt_factor, np.array(input_pts))
        python_result = np.sqrt(ca2_from_python) * kt_from_python
        metric = metric_dict.get(kt_factor, None)
        # First test the pairwise version
        for row in range(n_particles):
            for col in range(row):
                cpp_distance = CALE.GeneralisedKtDistance2(input_pts[row],
                                                           input_rapidities[row],
                                                           input_phis[row],
                                                           input_pts[col],
                                                           input_rapidities[col],
                                                           input_phis[col],
                                                           kt_factor)
                cpp_distance = np.sqrt(cpp_distance)
                python_distance = python_result[row, col]
                error_message = (f"kt_factor = {kt_factor}, pt_row = {input_pts[row]}, "
                                 + f"pt_col = {input_pts[col]}, "
                                 + f"rapidity_row = {input_rapidities[row]}, "
                                 + f"rapidity_col = {input_rapidities[col]}, "
                                 + f"phi_row = {input_phis[row]}, phi_col = {input_phis[col]}"
                                 + f"python_result = {python_distance}, cpp_result = ")
                tst.assert_allclose(cpp_distance, python_distance,
                                    err_msg = error_message+str(cpp_distance))
                if metric is not None:
                    cpp_distance = CALE.NamedDistance2(input_pts[row],
                                                       input_rapidities[row],
                                                       input_phis[row],
                                                       input_pts[col],
                                                       input_rapidities[col],
                                                       input_phis[col],
                                                       metric)
                    cpp_distance = np.sqrt(cpp_distance)
                    tst.assert_allclose(cpp_distance, python_distance,
                                        err_msg = error_message+str(cpp_distance))
        # Now test the matrix version
        cpp_result = CALE.GeneralisedKtDistance2Matrix(input_pts,
                                                       input_rapidities,
                                                       input_phis,
                                                       kt_factor)
        cpp_result = np.sqrt(cpp_result)

        # generally don't care if there are nans on the diagonal that don't match
        for i in range(n_particles):
            if np.isnan(python_result[i,i]):
                cpp_result[i][i] = np.nan
        tst.assert_allclose(python_result, cpp_result)
        if metric is not None:
            cpp_result = CALE.NamedDistance2Matrix(input_pts,
                                                   input_rapidities,
                                                   input_phis,
                                                   metric)
            cpp_result = np.sqrt(cpp_result)
            for i in range(n_particles):
                if np.isnan(python_result[i,i]):
                    cpp_result[i][i] = np.nan
            tst.assert_allclose(cpp_result, python_result)



class TestCPPExpAffinity(test_FormJets.TestExpAffinity):
    def function(self, distances2, sigma=1, exponant=2, fill_diagonal=True):
        if exponant != 2:
            raise NotImplementedError("Can't do that in c++")
        if not fill_diagonal:
            raise NotImplementedError("Can't avoid that in c++")
        if hasattr(distances2, "tolist"):
            distances2 = distances2.tolist()
        return CALE.Affinities(distances2, sigma)

    def test_assymetric(self):
        pass  # disable these tests as they are not implemented in c++

    def test_exponent_fill_diagonal(self):
        pass  # disable these tests as they are not implemented in c++


def test_Laplacian():
    # shouldn't choke on an empty matrix
    input_distances2 = []
    sigma = 1
    normalised = True
    found = CALE.Laplacian(input_distances2, sigma, normalised)
    found = np.array(found)
    assert len(found.flatten()) == 0
    # can't really import the existing test as it starts from an
    # affinity matrix, but we can compare outputs with the python version
    distance2_matrices = ([[0., 1.], [1., 0.]],
                          [[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]],
                          [[1., 2., 3.], [2., 1., 2.], [3., 2., 1.]])
    sigmas = [1, 2, 3]
    for input_distances2 in distance2_matrices:
        for sigma in sigmas:
            exp_affinity = FormJets.exp_affinity(np.array(input_distances2), sigma)
            found = CALE.Laplacian(input_distances2, sigma, True)
            python_version = FormJets.symmetric_laplacian(exp_affinity)
            tst.assert_allclose(found, python_version)
            found = CALE.Laplacian(input_distances2, sigma, False)
            python_version = FormJets.unnormed_laplacian(exp_affinity)
            tst.assert_allclose(found, python_version)



def test_PxPyPz_PtRapPhi():
    atol = 1e-5
    # With no transverse momentum, phi isn't well defined.
    energy = 1
    ptrapphi = (0, 0, 0)
    pxpypz = (0, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found, pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found[:2], ptrapphi[:2], atol=atol)
    energy = 1
    ptrapphi = (0, 0, 1)
    pxpypz = (0, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found[:2], ptrapphi[:2], atol=atol)
    energy = 1
    ptrapphi = (1, 0, 0)
    pxpypz = (1, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found[:2], ptrapphi[:2], atol=atol)
    energy = 1
    ptrapphi = (1, 0, np.pi/2.)
    pxpypz = (0, 1, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 2
    ptrapphi = (1, 0, 0)
    pxpypz = (1, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 3
    ptrapphi = (2, 0, 0)
    pxpypz = (2, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, 0, -np.pi/2)
    pxpypz = (0, -1, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found, pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, 0, np.pi)
    pxpypz = (-1, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, 0, 2*np.pi)
    pxpypz = (1, 0, 0)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    # 2pi is 0
    tst.assert_allclose(found, [1, 0, 0], atol=atol)
    energy = 1
    ptrapphi = (1, 1, 0)
    pz = 1*(np.exp(2*1) - 1)/(np.exp(2*1) + 1)
    pxpypz = (1, 0, pz)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found, pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 5
    ptrapphi = (1, 1, 0)
    pz = 5*(np.exp(2*1) - 1)/(np.exp(2*1) + 1)
    pxpypz = (1, 0, pz)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, 4, 0)
    pz = 1*(np.exp(2*4) - 1)/(np.exp(2*4) + 1)
    pxpypz = (1, 0, pz)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, -4, np.pi/4)
    pz = -1*(np.exp(2*4) - 1)/(np.exp(2*4) + 1)
    pxpypz = (1/np.sqrt(2), 1/np.sqrt(2), pz)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)
    energy = 1
    ptrapphi = (1, -4, 3*np.pi/4)
    pz = -1*(np.exp(2*4) - 1)/(np.exp(2*4) + 1)
    pxpypz = (-1/np.sqrt(2), 1/np.sqrt(2), pz)
    found = CALE.PxPyPz(energy, *ptrapphi)
    tst.assert_allclose(found,pxpypz, atol=atol)
    found = CALE.PtRapPhi(energy, *pxpypz)
    tst.assert_allclose(found, ptrapphi, atol=atol)


