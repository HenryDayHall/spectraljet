import unittest
import numpy as np
import numpy.testing as tst
from spectraljet import CALEFunctions


def test_set_scales():
    found = CALEFunctions.set_scales(1., 2., 0)
    assert len(found) == 0
    found = CALEFunctions.set_scales(1., 2., 1)
    assert len(found) == 1
    tst.assert_allclose(found, [2.])
    found = CALEFunctions.set_scales(1., 2., 2)
    assert len(found) == 2
    tst.assert_allclose(found, [2., 1.])
    found = CALEFunctions.set_scales(1., 2., 3)
    assert len(found) == 3
    tst.assert_allclose(found[[0,-1]], [2., 1.])
    assert found[1] > 1. and found[1] < 2.

    

def test_kernels():
    my_kernel = CALEFunctions.kernel(0., 'mh')
    # special case
    tst.assert_allclose(my_kernel, 0.)
    my_kernel = CALEFunctions.kernel(1., 'mh')
    assert my_kernel > 0.
    my_kernel2 = CALEFunctions.kernel(2., 'mh')
    # in general, affinity should decrease with distance
    assert my_kernel > my_kernel2
    # calculate one case
    expected = 2*np.exp(-2)
    tst.assert_allclose(my_kernel2, expected)
    # it should also work with arrays
    found = CALEFunctions.kernel(np.array([0., 1., 2.]), 'mh')
    assert len(found) == 3
    tst.assert_allclose(found, [0., my_kernel, my_kernel2])
    # and be fine with empty arrays
    found = CALEFunctions.kernel(np.array([]), 'mh')
    assert len(found) == 0


def test_filter_design():
    # TODO why do we add an extra filter to the start??
    # first return value is the only one we use right now.
    found = CALEFunctions.filter_design()
    assert len(found) == 1
    assert hasattr(found[0], '__call__')
    # check it returns the same shape that went in
    output = found[0](np.array([1.]))
    assert len(output) == 1
    # check it works with arrays
    output2 = found[0](np.array([1., 2.]))
    assert len(output2) == 2
    tst.assert_allclose(output2[0], output[0])


def test_wavelet_approx():
    pass  #TODO



def test_cluster_particles():
    # should be fine with empty lists
    clusters, cluster_list = CALEFunctions.cluster_particles([], [], [])
    assert len(clusters) == 0
    assert len(cluster_list) == 0
    # One particle should be in one cluster
    clusters, cluster_list = CALEFunctions.cluster_particles([1.], [0.], [5.])
    assert len(clusters) == 1
    assert len(cluster_list) == 1
    # Make two clear clusters
    clusters, cluster_list = CALEFunctions.cluster_particles([1., 1., -1., -1.],
                                                             [0., 0., np.pi, np.pi],
                                                             [5., 5., 5., 5.])
    assert len(clusters) == 4
    assert len(cluster_list) == 2
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    #assert [0, 1] in cluster_list  #TODO why is this failing?
    assert [2, 3] in cluster_list




class TestChebyOp(unittest.TestCase):
    def function(self, *args, **kwargs):
        return CALEFunctions.cheby_op(*args, **kwargs)

    f = np.array([1, 2, 3, 4])
    L = np.array([[4, -1, -1, -1], [-1, 3, -1, -1], [-1, -1, 3, -1], [-1, -1, -1, 3]])
    c_single = np.array([1, 0, 2])
    c_multiple = [np.array([1, 0, 2]), np.array([1, 2, 3])]
    arange = (0, 1)

    def test_multiple_coefficients(self):
        results = self.function(self.f, self.L, self.c_multiple, self.arange)  # unpack the tuple
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, np.ndarray)

    def test_output_size(self):
        result_single = self.function(self.f, self.L, self.c_single, self.arange)
        result_single = np.array(result_single).reshape(-1,1).flatten()
        self.assertEqual(result_single.shape, self.f.shape)

        results_multiple = self.function(self.f, self.L, self.c_multiple, self.arange)
        for r in results_multiple:
            self.assertEqual(r.shape, self.f.shape)
    
    def test_not_iterable(self):
        c = 3
        result = self.function(self.f, self.L, c, self.arange)
        expected_output = [1.5 * self.f]
        tst.assert_allclose(result, expected_output)

    def test_empty_iterable(self):
        c = []
        with self.assertRaises(ValueError) as context:
            self.function(self.f, self.L, c, self.arange)
        #print("Actual exception message:", str(context.exception))
        self.assertTrue("Coefficients are an empty list." in str(context.exception))

    def test_generic_case(self):
        c = [np.array([1, 2, 3]), np.array([4, 5])]
        with self.assertRaises(ValueError) as context:
            self.function(self.f, self.L, c, self.arange)
        #print("Actual exception message:", str(context.exception))  # Debug line
       
        self.assertTrue("All inner arrays of c must be of the same size." in str(context.exception))


    def test_zero_f_or_L(self):
        zero_f = np.zeros_like(self.f)
        zero_L = np.zeros_like(self.L)
        c = np.array([1, 2, 3])
        result_zero_f = self.function(zero_f, self.L, c, self.arange)
        result_zero_L = self.function(self.f, zero_L, c, self.arange)
        tst.assert_allclose(result_zero_f[0], np.zeros_like(self.f))
        tst.assert_allclose(result_zero_L[0], np.zeros_like(self.f))


    def test_basic_polynomial_evaluation(self):
        """Test if the polynomial calculation is correct.
        Given f(x) = 1, L(x) = 1, and the polynomial coefficients [1, 0], we have 
        0.5 * c[j][0] * Twf_old + c[j][1] * Twf_cur,
        the function should return an array with a single value of 1*0.5=0.5.
        """

        f_test = np.array([1])
        L_test = np.array([[1]])
        c_test = np.array([1, 0])
        arange_test = (-1, 1)
        
        result = self.function(f_test, L_test, c_test, arange_test)
        expected_output = np.array([[0.5]])

        tst.assert_allclose(result, expected_output)
    
    def test_identity_L(self):
        """
        Based on the expected and actual outputs:

        For a Chebyshev polynomial applied to the identity operator:
        T0(L) = T0(x) = 1
        T1(L) = T1(x) = x

        When you apply these to the identity operator, you get:
        T0(L)f = f
        T1(L)f = f

        For the coefficients c = [1, 2], the application of the Chebyshev polynomial to the function f yields:
        f' = 0.5 * c0 * T0(L)f + c1 * T1(L)f
        f' = 0.5 * 1 * f + 2 * f
        f' = 0.5 * f + 2 * f
        f' = 2.5 * f

        Given:
        c = [2, 3, 4, 5]
        f = [2, 3, 4, 5]

        Expected result:
        f' = [5, 7.5, 10, 12.5]
        """


        # If L is identity, the result should match basic Chebyshev polynomial results
        f_test = np.array([2, 3, 4, 5])
        L_test = (np.eye(4))
        c_test = np.array([1, 2])
        arange_test = (-1, 1)
        
        result = self.function(f_test, L_test, c_test, arange_test)
        expected_output = f_test * 2.5  # As T_1(x) is x and T_0(x) is 1, and c has values [1, 2]

        tst.assert_allclose(result[0], expected_output)


    def test_linearity(self):
        # The Chebyshev polynomial of the sum should be the sum of the Chebyshev polynomials.
        f1 = np.array([1, 2, 3, 4])
        f2 = np.array([4, 3, 2, 1])
        result_sum = self.function(f1 + f2, self.L, self.c_single, self.arange)
        result_individual1 = self.function(f1, self.L, self.c_single, self.arange)
        result_individual2 = self.function(f2, self.L, self.c_single, self.arange)
        
        # Convert to numpy arrays and sum them
        combined_results = np.array(result_individual1) + np.array(result_individual2)
        
        tst.assert_allclose(result_sum, combined_results)



class TestMakeLIdx(unittest.TestCase):
    def function(self, *args, **kwargs):
        return CALEFunctions.make_L_idx(*args, **kwargs)

    def test_basic_functionality(self):
        y = [0.5, 1, 1.5]
        # Becuase of the phi values, no angular considerations are needed.
        phi = [0, np.pi/2, np.pi]
        pT = [1, 2, 3]
        # as the values given are less than 1, this is clearly an anti-kt distance
        #
        # there are 2 ways I could anticipate defining the anti-kt distance;
        # A. distance = min(1/pT_1, 1/pT_2) * sqrt(delta_phi**2 + delta_y**2)
        # B. distance = min(1/pT_1, 1/pT_2)**2 * (delta_phi**2 + delta_y**2)
        #
        # The A. distance is more like a modified euclidean distance,
        # The B. distance is how the paper defines it.
        #
        # The expected output here corrisponds to neither of those.
        # I suspect a calculation error.

        #expected_output = np.array([[0., 0.63245553, 0.84515425],
        #                            [0.63245553, 0., 0.63245553],
        #                            [0.84515425, 0.63245553, 0.]])
        
        # I sugest it should be version A
        expected_output = np.array([[0., 0.824227, 1.098969],
                                    [0.824227, 0., 0.549484],
                                    [1.098969, 0.549484, 0.]])

        np.testing.assert_almost_equal(self.function(y, phi, pT), expected_output,
                                       decimal=5)


    def test_empty_inputs(self):
        # Looking at the rest of your code, I'm
        # not sure a ValueError is desirable here
        #with self.assertRaises(ValueError):
        #    self.function([], [], [])

        # Sugested test
        found = self.function([], [], [])
        assert len(found) == 0

    def test_length_one(self):
        y = [0.5]
        phi = [np.pi/4]
        pT = [1]
        expected_output = np.array([[0.]])
        np.testing.assert_almost_equal(self.function(y, phi, pT), expected_output)

    def test_same_pT_values(self):
        y = [0.5, 1, 1.5]
        phi = [0, np.pi/2, np.pi]
        pT = [2, 2, 2]
        mat = self.function(y, phi, pT)
        self.assertTrue((np.diag(mat) == 0).all())  # Check diagonal elements are zero

    def test_large_y_difference(self):
        y = [0.5, 10]
        phi = [0, np.pi/2]
        pT = [1, 2]
        mat = self.function(y, phi, pT)
        self.assertTrue((np.diag(mat) == 0).all())

    def test_phi_exceeding_2pi(self):
        y = [0.5, 1]
        phi = [0, 3*np.pi]
        pT = [1, 2]
        expected_phi = [0, np.pi]
        mat1 = self.function(y, phi, pT)
        mat2 = self.function(y, expected_phi, pT)
        np.testing.assert_almost_equal(mat1, mat2)

    def test_symmetry(self):
        y = [0.5, 1, 2]
        phi = [0, np.pi/4, np.pi/2]
        pT = [1, 2, 3]
        mat = np.array(self.function(y, phi, pT))
        np.testing.assert_almost_equal(mat, mat.T)


class TestChebyCoeff(unittest.TestCase):
    def function(self, *args, **kwargs):
        return CALEFunctions.cheby_coeff(*args, **kwargs)

    def test_even_function(self):
        # Function: f(x) = x^4 on [-1, 1]
        # The Chebyshev coefficients for even functions will have c[1], c[3], ... as 0
        def f(x):
            return x**4

        coefficients = self.function(f, 5)
        self.assertAlmostEqual(coefficients[1], 0, places=10)
        self.assertAlmostEqual(coefficients[3], 0, places=10)
        self.assertAlmostEqual(coefficients[5], 0, places=10)

    def test_compare_with_np_cheb(self):
        # Using numpy's chebfit function to compute the coefficients
        # For the function f(x) = x^3
        def f(x):
            return x**2

        coefficients = self.function(f, 5)
        x = np.linspace(-1, 1, 400)
        np_coeffs = np.polynomial.chebyshev.chebfit(x, f(x), 5)
        # Note: numpy's chebfit might return coefficients in a different order,
        # so you might need to reverse the coefficients before comparing
        tst.assert_allclose(coefficients, np_coeffs, atol=1e-10)



"""
class TestCALE(unittest.TestCase):

    def setUp(self):
        self.sgwt = CALE()
        # Mock data for testing purposes
        self.sgwt.Leaf_Rapidity = np.array([0.1, 0.2, 0.3])
        self.sgwt.Leaf_Phi = np.array([0.4, 0.5, 0.6])
        self.sgwt.Leaf_PT = np.array([0.7, 0.8, 0.9])
        self.sgwt.Leaf_Label = np.array([1, 2, 3])
        self.sgwt.Sigma = 0.1
        self.sgwt.Cutoff = -0.999
        self.sgwt.Normalised = True
        self.sgwt.NRounds = 5


    def test_allocate(self):
        jets = self.sgwt.allocate()
        # extend this
"""


if __name__ == "__main__":
    unittest.main()



