import unittest
import numpy as np
import numpy.testing as tst
from spectraljet import CALEFormJets
from . import test_FormJets


np.random.seed(42)
n_particles = 10
complex_energies = np.random.rand(n_particles)*10. + 2.
complex_pxs = np.random.rand(n_particles)*2. - 1.
complex_pys = np.random.rand(n_particles)*2. - 1.
complex_pzs = np.random.rand(n_particles)*2. - 1.

class TestCALE(unittest.TestCase):
    to_test = CALEFormJets.CALE
    def test_empty(self):
        # Do a few basic clustering to check that logical things happen
        # in clear cut cases.
        # ~~~~~~~~
        # shouldn't get and error in empty input
        energies = np.array([])
        pxs = np.array([])
        pys = np.array([])
        pzs = np.array([])
        algo = self.to_test.from_kinematics(energies, pxs, pys, pzs,
                                            dict_jet_params = dict(NRounds=15, Sigma=0.1, Cutoff=0.))
        algo.run()
        jets = algo.split()
        assert len(jets) == 0

    def test_one_particle(self):
        # ~~~~~~~~
        # With a single particle, we should get a single jet
        energies = np.array([100.])
        pxs = np.array([0.])
        pys = np.array([1.])
        pzs = np.array([0.])
        algo = self.to_test.from_kinematics(energies, pxs, pys, pzs,
                                            dict_jet_params = dict(NRounds=15, Sigma=0.1, Cutoff=0.))
        algo.run()
        jets = algo.split()
        assert len(jets) == 1
        # we expect this jet to have one particle and one root
        mask = jets[0].Label != -1
        assert sum(mask) == 2
        assert sum(jets[0].Parent[mask] == -1) == 1

    def test_trivial_case(self):
        # ~~~~~~~~
        # With four particles in two clear clusters, we should get two jets
        energies = np.array([100., 100., 100., 100.])
        pxs = np.array([0., 0., 1., 1.])
        pys = np.array([1., 1., 0., 0.])
        pzs = np.array([-1., -1., 1., 1.])
        algo = self.to_test.from_kinematics(energies, pxs, pys, pzs,
                                            dict_jet_params = dict(NRounds=15, Sigma=0.1, Cutoff=0.))
        algo.run()
        #TODO currently, this missbehaves; i.e. it forms one jet and two single particles
        jets = algo.split()
        print(jets)
        #assert len(jets) == 2
        ## Each jet should have two particles and one root
        #mask = jets[0].Label != -1
        #assert len(jets[0].Parent[mask]) == 3
        #assert sum(jets[0].Parent[mask] == -1) == 1
        #mask = jets[1].Label != -1
        #assert len(jets[1].Parent[mask]) == 3
        #assert sum(jets[1].Parent[mask] == -1) == 1
        ## The first two particles should be in one jet, the second two in the other
        #non_root_idxs_0 = jets[0].Label[jets[0].Parent != -1]
        #assert abs(non_root_idxs_0[0] - non_root_idxs_0[0]) == 1
        #non_root_idxs_1 = jets[1].Label[jets[1].Parent != -1]
        #assert abs(non_root_idxs_1[1] - non_root_idxs_1[1]) == 1
        ## The jets should both have 200. energy
        #tst.assert_allclose(jets[0].E[(jets[0].Label != -1)*(jets[0].Parent == -1)], 200.)
        #tst.assert_allclose(jets[1].E[(jets[1].Label != -1)*(jets[1].Parent == -1)], 200.)
        return algo

    def test_CALE_complex(self):
        # this test is expected to break if the algorithm is changed
        algo = self.to_test.from_kinematics(complex_energies, complex_pxs, complex_pys, complex_pzs,
                                            dict_jet_params = dict(NRounds=15, Sigma=0.1, Cutoff=0.))
        algo.run()
        ints = np.array([[0, 10, -1, -1, -1],
                         [1, 10, -1, -1, -1],
                         [2, 10, -1, -1, -1],
                         [3, 10, -1, -1, -1],
                         [4, 10, -1, -1, -1],
                         [5, 10, -1, -1, -1],
                         [6, 11, -1, -1, -1],
                         [7, 10, -1, -1, -1],
                         [8, 10, -1, -1, -1],
                         [9, 12, -1, -1, -1],
                         [10, -1, 10, 10, 0],
                         [11, -1, 11, 11, 0],
                         [12, -1, 12, 12, 0]])
                         
        floats = np.array([[9.845817328538798430e-01, 3.745435093576598984e-02, 2.912381932908990834e+00, 5.745401188473625353e+00, -9.588310114083951063e-01, 2.237057894447589401e-01, 2.150897038028767305e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [1.184533571776632765e+00, -5.732730138169838535e-02, -6.544068836423426738e-01, 1.150714306409916077e+01, 9.398197043239886472e-01, -7.210122786959163310e-01, -6.589517526254169422e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [7.841478344162055025e-01, -9.360964620855077856e-02, -5.587695339728266930e-01, 9.319939418114049801e+00, 6.648852816008434807e-01, -4.157107029295636913e-01, -8.698968140294409679e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [6.343751072456259577e-01, 1.128869759878429191e-01, -2.706689914661974949e+00, 7.986584841970366000e+00, -5.753217786434476899e-01, -2.672763134126165951e-01, 8.977710745066664888e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [6.423867924528151585e-01, 2.678008446756039995e-01, -3.004391378345296015e+00, 3.560186404424364959e+00, -6.363500655857987631e-01, -8.786003156592814278e-02, 9.312640661491187188e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [8.521925447633863504e-01, 1.750252026237112679e-01, 2.408358853795733445e+00, 3.559945203362026689e+00, -6.331909802931323661e-01, 5.703519227860271990e-01, 6.167946962329222682e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [7.169851785485583662e-01, -1.525864024174706235e-01, -2.148447868759928880e+00, 2.580836121681994832e+00, -3.915155140809245538e-01, -6.006524356832805278e-01, -3.907724616532586293e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [5.711392629166963525e-02, -7.561496886439775245e-02, 5.218188045310787615e-01, 1.066176145774935158e+01, 4.951286326447568165e-02, 2.846887682722321067e-02, -8.046557719872322600e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [2.295380842089535245e-01, 4.602662579251593639e-02, 2.205542730943898633e+00, 8.011150117432087825e+00, -1.361099627157684733e-01, 1.848291377240849354e-01, 3.684660530243137888e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [9.985839975171026950e-01, -1.318197997449831009e-02, -2.002186256784725504e+00, 9.080725777960454437e+00, -4.175417196039161727e-01, -9.070991745600045508e-01, -1.196950125207973947e-01, 0.000000000000000000e+00, 1.000000000000000000e+00],
                           [1.373854057635769621e+00, 1.153086572956853878e-02, -2.781180254985658351e+00, 6.035211169562503386e+01, -1.285585949457234589e+00, -4.845035998219304751e-01, 6.958812550738078251e-01, np.nan, 8.000000000000000000e+00],
                           [7.169851785485583662e-01, -1.525864024174706235e-01, -2.148447868759928880e+00, 2.580836121681994832e+00, -3.915155140809245538e-01, -6.006524356832805278e-01, -3.907724616532586293e-01, np.nan, 1.000000000000000000e+00],
                           [9.985839975171026950e-01, -1.318197997449831009e-02, -2.002186256784725504e+00, 9.080725777960454437e+00, -4.175417196039161727e-01, -9.070991745600045508e-01, -1.196950125207973947e-01, np.nan, 1.000000000000000000e+00]])
        test_FormJets.match_ints_floats(ints, floats,
                                        algo._ints[:len(ints)], algo._floats[:len(floats)])

# TODO, kernal used fro the chebyshev coefficients needs
# to be the absspline kernal for this to work.
#
#class TestCALECpp(TestCALE):
#    to_test = CALEFormJets.CALECpp
