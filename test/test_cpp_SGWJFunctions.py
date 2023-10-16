import numpy as np
from numpy import testing as tst
from spectraljet import Constants, FormJets
from spectraljet.cpp_sgwj import build

build_dir = Constants.sgwj_build_dir
build.build(build_dir, force_rebuild = False)
sgwj = build.get_module(build_dir)

from . import test_SGWTFunctions, test_Components, test_FormJets


#class TestCPPChebyCoeff(test_SGWTFunctions.TestChebyCoeff):
#    def function(self, g, m, N=None, arange=(-1,1)):
#        return sgwj.ChebyshevCoefficients(m, N, arange[0], arange[1])
#
#
#class TestCPPMakeLIdx(test_SGWTFunctions.TestMakeLIdx):
#    def function(self, particle_rapidities, particle_phis, particle_pts):
#        particle_rapidities = list(particle_rapidities)
#        particle_phis = list(particle_phis)
#        particle_pts = list(particle_pts)
#        metric = sgwj.JetMetrics.antikt
#        return sgwj.NamedDistanceMetric(particle_pts, particle_rapidities, particle_phis, metric)
#
#
#
#class TestCPPChebyOp(test_SGWTFunctions.TestChebyOp):
#    def function(self, wavelet_delta, laplacian, chebyshef_coefficients, arange):
#        laplacian = list(list(row) for row in laplacian)
#        chebyshef_coefficients = list(chebyshef_coefficients)
#        center_idx = np.where(wavelet_delta)[0][0]
#        interval = list(arange)
#        return sgwj.LaplacianWavelet(laplacian, chebyshef_coefficients, center_idx, interval)


def test_VectorAddition():
    # Two empty vectors should make another empty vector
    assert len(sgwj.VectorAddition([], [])) == 0
    # Two vectors with 0 in should make another vectors with 0 in
    tst.assert_allclose(sgwj.VectorAddition([0,0,0], [0,0,0]), [0,0,0])
    # Two vectors with 1 in should make another vectors with 2 in
    tst.assert_allclose(sgwj.VectorAddition([1,1,1], [1,1,1]), [2,2,2])
    # A vector of -1 and a vector of 1 should make a vector of 0
    tst.assert_allclose(sgwj.VectorAddition([-1,-1,-1], [1,1,1]), [0,0,0])
    # One last test
    tst.assert_allclose(sgwj.VectorAddition([1,2,3], [4,5,6]), [5,7,9])


def test_MatrixDotVector():
    # Empty matrix and empty vector should make an empty vector
    assert len(sgwj.MatrixDotVector([], [])) == 0
    # Length one matrix and length one vector should make a length one vector
    tst.assert_allclose(sgwj.MatrixDotVector([[1]], [1]), [1])
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
            tst.assert_allclose(sgwj.MatrixDotVector(matrix, vector), np.dot(matrix, vector))
    # again but length 3
    length_3_matrices = ([[0,0,0],[0,0,0],[0,0,0]],
                         [[1,2,3],[4,5,6],[7,8,9]])
    length_3_vectors = ([0,0,0],
                        [1,2,3])
    # Test all combinations of the above against numpy
    for matrix in length_3_matrices:
        for vector in length_3_vectors:
            tst.assert_allclose(sgwj.MatrixDotVector(matrix, vector), np.dot(matrix, vector))



class TestCPPAngularDistance(test_Components.TestAngularDistance):
    def function(self, a, b):
        if hasattr(a, '__iter__'):
            return [sgwj.AngularDistance(a_i, b_i) for a_i, b_i in zip(a, b)]
        return sgwj.AngularDistance(a, b)


def test_CambridgeAachenDistance2():
    # Too complicated to import existing test due to matrix structures;
    # just test a few cases
    found = sgwj.CambridgeAachenDistance2(0, 0, 0, 0)
    tst.assert_allclose(found, 0)
    found = sgwj.CambridgeAachenDistance2(1, 0, 0, 0)
    tst.assert_allclose(found, 1)
    found = sgwj.CambridgeAachenDistance2(0, 1, 0, 0)
    tst.assert_allclose(found, 1)
    found = sgwj.CambridgeAachenDistance2(0, 0, 1, 0)
    tst.assert_allclose(found, 1)
    found = sgwj.CambridgeAachenDistance2(0, 0, 0, 1)
    tst.assert_allclose(found, 1)
    found = sgwj.CambridgeAachenDistance2(1, 1, 0, 0)
    tst.assert_allclose(found, 2)
    found = sgwj.CambridgeAachenDistance2(1, 0, 1, 0)
    tst.assert_allclose(found, 0)
    found = sgwj.CambridgeAachenDistance2(1, 0, 0, 1)
    tst.assert_allclose(found, 2)
    found = sgwj.CambridgeAachenDistance2(2.5, 0, 0, 0)
    tst.assert_allclose(found, 6.25)
    found = sgwj.CambridgeAachenDistance2(0, -2.5, 0, 0)
    tst.assert_allclose(found, 6.25)
    found = sgwj.CambridgeAachenDistance2(0, np.pi, 0, 1.-np.pi)
    tst.assert_allclose(found, 1.)


def test_GeneralisedKtDistance():
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
    metric_dict = {-1: sgwj.JetMetrics.antikt, 0: sgwj.JetMetrics.cambridge_aachen, 1: sgwj.JetMetrics.kt}
    for kt_factor in [-1, 0, 0.5, 1]:
        kt_from_python = FormJets.genkt_factor(kt_factor, np.array(input_pts))
        python_result = np.sqrt(ca2_from_python) * kt_from_python
        metric = metric_dict.get(kt_factor, None)
        # First test the pairwise version
        for row in range(n_particles):
            for col in range(row):
                cpp_distance = sgwj.GeneralisedKtDistance(input_pts[row],
                                                          input_rapidities[row],
                                                          input_phis[row],
                                                          input_pts[col],
                                                          input_rapidities[col],
                                                          input_phis[col],
                                                          kt_factor)
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
                    cpp_distance = sgwj.NamedDistance(input_pts[row],
                                                      input_rapidities[row],
                                                      input_phis[row],
                                                      input_pts[col],
                                                      input_rapidities[col],
                                                      input_phis[col],
                                                      metric)
                    tst.assert_allclose(cpp_distance, python_distance,
                                        err_msg = error_message+str(cpp_distance))
        # Now test the matrix version
        cpp_result = sgwj.GeneralisedKtDistanceMatrix(input_pts,
                                                      input_rapidities,
                                                      input_phis,
                                                      kt_factor)

        # generally don't care if there are nans on the diagonal that don't match
        for i in range(n_particles):
            if np.isnan(python_result[i,i]):
                cpp_result[i][i] = np.nan
        tst.assert_allclose(python_result, cpp_result)
        if metric is not None:
            cpp_result = sgwj.NamedDistanceMatrix(input_pts,
                                                  input_rapidities,
                                                  input_phis,
                                                  metric)
            for i in range(n_particles):
                if np.isnan(python_result[i,i]):
                    cpp_result[i][i] = np.nan
            tst.assert_allclose(cpp_result, python_result)



def test_Affinities():
    pass
    #TODO import test and apply

def test_Laplacian():
    pass
    #TODO import test and apply

def test_PxPyPz():
    pass
    #TODO import test and apply

def test_PtRapPhi():
    pass
    #TODO import test and apply

