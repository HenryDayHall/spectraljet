import numpy as np
from spectraljet import Constants
from spectraljet.cpp_sgwj import build

build_dir = Constants.sgwj_build_dir
build.build(build_dir, force_rebuild = True)
sgwj = build.get_module(build_dir)

from . import test_SGWTFunctions


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
    pass
    #TODO


def test_MatrixDotVector():
    pass
    #TODO


def test_AngularDistance():
    pass
    #TODO import test and apply

def test_CambridgeAachen2Distance():
    pass
    #TODO import test and apply

def test_GeneralisedKTDistance():
    pass
    #TODO import test and apply

def test_NamedDistance():
    pass
    #TODO import test and apply

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

