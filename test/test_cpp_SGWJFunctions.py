import numpy as np
from spectraljet import Constants
from spectraljet.cpp_sgwj import build

build_dir = Constants.sgwj_build_dir
build.build(build_dir, force_rebuild = True)
sgwj = build.load(build_dir)

from . import test_SGWTFunctions

def cpp_cheby_coeff(g, m, N=None, arange=(-1,1)):
    return sgwj.ChebyshevCoefficients(m, N, arange[0], arange[1])


class TestCPPChebyCoeff(test_SGWTFunctions.TestChebyCoeff):
    function_to_test = cpp_cheby_coeff

def cpp_make_L_idx(particle_rapidities, particle_phis, particle_pts):
    particle_rapidities = list(particle_rapidities)
    particle_phis = list(particle_phis)
    particle_pts = list(particle_pts)
    metric = sgwj.JetMetrics.antikt
    return sgwj.NamedDistanceMetric(particle_pts, particle_rapidities, particle_phis, metric)

class TestCPPMakeLIdx(test_SGWTFunctions.TestMakeLIdx):
    function_to_test = cpp_make_L_idx


def cpp_cheby_op(wavelet_delta, laplacian, chebyshef_coefficients, arange):
    laplacian = list(list(row) for row in laplacian)
    chebyshef_coefficients = list(chebyshef_coefficients)
    center_idx = np.where(wavelet_delta)[0][0]
    interval = list(arange)
    return sgwj.LaplacianWavelet(laplacian, chebyshef_coefficients, center_idx, interval)

class TestCPPChebyOp(test_SGWTFunctions.TestChebyOp):
    function_to_test = cpp_cheby_op


def test_VectorAddition():
    #TODO


def test_MatrixDotVector():
    #TODO


def test_AngularDistance():
    #TODO import test and apply

def test_CambridgeAachen2Distance():
    #TODO import test and apply

def test_GeneralisedKTDistance():
    #TODO import test and apply

def test_NamedDistance():
    #TODO import test and apply

def test_Affinities():
    #TODO import test and apply

def test_Laplacian():
    #TODO import test and apply

def test_PxPyPz():
    #TODO import test and apply

def test_PtRapPhi():
    #TODO import test and apply

