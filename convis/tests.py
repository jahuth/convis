import unittest


class TestConvis(unittest.TestCase):
    def test_retina(self):
        from . import retina
        import numpy as np
        ret = retina.Retina()
        stimulus = np.zeros((5000,5,5))
        T,X,Y = np.meshgrid(np.arange(stimulus.shape[0]),
                            np.arange(stimulus.shape[1]),
                            np.arange(stimulus.shape[2]),indexing='ij')
        stimulus += 0.5*(1.0+np.sin(0.1 * T + 0.5*X + 0.3*Y))
        o = ret.run_in_chunks(stimulus, 200)
        self.assertEqual(len(o),2)

class TestConvisFilters(unittest.TestCase):
    def test_exponentials(self):
        from . import variables
        from . import numerical_filters
        import numpy as np
        tau = 0.1
        resolution = variables.ResolutionInfo()
        f = numerical_filters.exponential_filter_1d(tau,n=0)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Exponential filter is not normalized.")
        self.assertAlmostEqual(np.sum(f),1.0,msg="Cascade filter normalization is unprecise.")
        f = numerical_filters.exponential_filter_1d(tau,n=4)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Cascade exponential filter is not normalized.")
        self.assertAlmostEqual(np.sum(f),1.0,msg="Cascade exponential normalization is unprecise.")
    def test_gaussians(self):
        from . import numerical_filters
        import numpy as np
        g = numerical_filters.gauss_filter_2d(2.0,2.0)
        self.assertTrue(np.abs(np.sum(g) - 1.0)<0.1,msg="Gauss filter is not normalized.")
        self.assertAlmostEqual(np.sum(g),1.0,msg="Gauss filter normalization is unprecise.")




if __name__ == '__main__':
    unittest.main()
