import sys
from pathlib import Path
path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))
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
    jet_params = dict(NRounds=15, Sigma=0.1, Cutoff=0.)
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
                                            dict_jet_params = self.jet_params)
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
                                            dict_jet_params = self.jet_params)
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
                                            dict_jet_params = self.jet_params)
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
        #assert abs(non_root_idxs_0[0] - non_root_idxs_0[1]) == 1
        #non_root_idxs_1 = jets[1].Label[jets[1].Parent != -1]
        #assert abs(non_root_idxs_1[0] - non_root_idxs_1[1]) == 1
        ## The jets should both have 200. energy
        #tst.assert_allclose(jets[0].E[(jets[0].Label != -1)*(jets[0].Parent == -1)], 200.)
        #tst.assert_allclose(jets[1].E[(jets[1].Label != -1)*(jets[1].Parent == -1)], 200.)
        return algo

    def test_complex(self):
        # this test is expected to break if the algorithm is changed
        algo = self.to_test.from_kinematics(complex_energies, complex_pxs, complex_pys, complex_pzs,
                                            dict_jet_params = self.jet_params)
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


class TestCALECpp(TestCALE):
    to_test = CALEFormJets.CALECpp


class TestCALEv2(TestCALE):
    to_test = CALEFormJets.CALEv2
    jet_params = dict(Sigma=0.8, Cutoff=0.9, WeightExponent=0., SeedGenerator='PtCenter')

    def test_trivial_case(self):
        # ~~~~~~~~
        # With four particles in two clear clusters, we should get two jets
        energies = np.array([100., 100., 100., 100.])
        pxs = np.array([0., 0., 1., 1.])
        pys = np.array([1., 1., 0., 0.])
        pzs = np.array([-1., -1., 1., 1.])
        algo = self.to_test.from_kinematics(energies, pxs, pys, pzs,
                                            dict_jet_params = self.jet_params)
        algo.run()
        jets = algo.split()
        assert len(jets) == 2
        # Each jet should have two particles and one root
        mask = jets[0].Label != -1
        assert len(jets[0].Parent[mask]) == 3
        assert sum(jets[0].Parent[mask] == -1) == 1
        mask = jets[1].Label != -1
        assert len(jets[1].Parent[mask]) == 3
        assert sum(jets[1].Parent[mask] == -1) == 1
        # The first two particles should be in one jet, the second two in the other
        non_root_idxs_0 = jets[0].Label[jets[0].Parent != -1]
        assert abs(non_root_idxs_0[0] - non_root_idxs_0[1]) == 1
        non_root_idxs_1 = jets[1].Label[jets[1].Parent != -1]
        assert abs(non_root_idxs_1[0] - non_root_idxs_1[1]) == 1
        # The jets should both have 200. energy
        tst.assert_allclose(jets[0].Energy[(jets[0].Label != -1)*(jets[0].Parent == -1)], 200.)
        tst.assert_allclose(jets[1].Energy[(jets[1].Label != -1)*(jets[1].Parent == -1)], 200.)

    def test_complex(self):
        # this test is expected to break if the algorithm is changed
        algo = self.to_test.from_kinematics(complex_energies, complex_pxs, complex_pys, complex_pzs,
                                            dict_jet_params = self.jet_params)
        algo.run()
        ints = np.array([[ 0, 12, -1, -1, -1],
                         [ 1, 11, -1, -1, -1],
                         [ 2, 11, -1, -1, -1],
                         [ 3, 10, -1, -1, -1],
                         [ 4, 13, -1, -1, -1],
                         [ 5, 12, -1, -1, -1],
                         [ 6, 10, -1, -1, -1],
                         [ 7, 14, -1, -1, -1],
                         [ 8, 15, -1, -1, -1],
                         [ 9, 10, -1, -1, -1],
                         [10, -1, 10, 10,  0],
                         [11, -1, 11, 11,  0],
                         [12, -1, 12, 12,  0],
                         [13, -1, 13, 13,  0],
                         [14, -1, 14, 14,  0],
                         [15, -1, 15, 15,  0]])
                         
        floats = np.array(
                 [[ 9.84581733e-01,  3.74543509e-02,  2.91238193e+00,  5.74540119e+00,
                   -9.58831011e-01,  2.23705789e-01,  2.15089704e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 1.18453357e+00, -5.73273014e-02, -6.54406884e-01,  1.15071431e+01,
                    9.39819704e-01, -7.21012279e-01, -6.58951753e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 7.84147834e-01, -9.36096462e-02, -5.58769534e-01,  9.31993942e+00,
                    6.64885282e-01, -4.15710703e-01, -8.69896814e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 6.34375107e-01,  1.12886976e-01, -2.70668991e+00,  7.98658484e+00,
                   -5.75321779e-01, -2.67276313e-01,  8.97771075e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 6.42386792e-01,  2.67800845e-01, -3.00439138e+00,  3.56018640e+00,
                   -6.36350066e-01, -8.78600316e-02,  9.31264066e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 8.52192545e-01,  1.75025203e-01,  2.40835885e+00,  3.55994520e+00,
                   -6.33190980e-01,  5.70351923e-01,  6.16794696e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 7.16985179e-01, -1.52586402e-01, -2.14844787e+00,  2.58083612e+00,
                   -3.91515514e-01, -6.00652436e-01, -3.90772462e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 5.71139263e-02, -7.56149689e-02,  5.21818805e-01,  1.06617615e+01,
                    4.95128633e-02,  2.84688768e-02, -8.04655772e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 2.29538084e-01,  4.60266258e-02,  2.20554273e+00,  8.01115012e+00,
                   -1.36109963e-01,  1.84829138e-01,  3.68466053e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 9.98583998e-01, -1.31819800e-02, -2.00218626e+00,  9.08072578e+00,
                   -4.17541720e-01, -9.07099175e-01, -1.19695013e-01,  0.00000000e+00,
                    1.00000000e+00],
                  [ 2.25105073e+00,  1.97145197e-02, -2.23317261e+00,  1.96481467e+01,
                   -1.38437901e+00, -1.77502792e+00,  3.87303600e-01,          np.nan,
                    3.00000000e+00],
                  [ 1.96652415e+00, -7.35390362e-02, -6.16320569e-01,  2.08270825e+01,
                    1.60470499e+00, -1.13672298e+00, -1.52884857e+00,          np.nan,
                    2.00000000e+00],
                  [ 1.77906202e+00,  8.96378513e-02,  2.67892706e+00,  9.30534639e+00,
                   -1.59202199e+00,  7.94057712e-01,  8.31884400e-01,          np.nan,
                    2.00000000e+00],
                  [ 6.42386792e-01,  2.67800845e-01, -3.00439138e+00,  3.56018640e+00,
                   -6.36350066e-01, -8.78600316e-02,  9.31264066e-01,          np.nan,
                    1.00000000e+00],
                  [ 5.71139263e-02, -7.56149689e-02,  5.21818805e-01,  1.06617615e+01,
                    4.95128633e-02,  2.84688768e-02, -8.04655772e-01,          np.nan,
                    1.00000000e+00],
                  [ 2.29538084e-01,  4.60266258e-02,  2.20554273e+00,  8.01115012e+00,
                   -1.36109963e-01,  1.84829138e-01,  3.68466053e-01,          np.nan,
                    1.00000000e+00]])
        test_FormJets.match_ints_floats(ints, floats,
                                        algo._ints[:len(ints)], algo._floats[:len(floats)])

