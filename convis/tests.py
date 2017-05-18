import unittest


class TestConvis(unittest.TestCase):
    def setUp(self):
        import convis
        self.convis = convis
    def test_retina(self):
        import convis
        import numpy as np
        ret = convis.retina.Retina()
        ret.debug = True
        stimulus = np.zeros((2000,50,50))
        T,X,Y = np.meshgrid(np.arange(stimulus.shape[0]),
                            np.arange(stimulus.shape[1]),
                            np.arange(stimulus.shape[2]),indexing='ij')
        stimulus += 0.5*(1.0+np.sin(0.1 * T + 0.5*X + 0.3*Y))
        o = ret.run_in_chunks(stimulus, 200)
        self.assertEqual(len(o),2)

class TestConvisFilters(unittest.TestCase):
    def test_exponentials(self):
        import convis
        import numpy as np
        tau = 0.1
        resolution = convis.variables.ResolutionInfo()
        f = convis.numerical_filters.exponential_filter_1d(tau,n=0)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Exponential filter is not normalized.")
        f = convis.numerical_filters.exponential_filter_1d(tau,n=4)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Cascade exponential filter is not normalized.")
    def test_gaussians(self):
        import convis
        import numpy as np
        g = convis.numerical_filters.gauss_filter_2d(2.0,2.0)
        self.assertTrue(np.abs(np.sum(g) - 1.0)<0.1,msg="Gauss filter is not normalized.")




if __name__ == '__main__':
    unittest.main()
