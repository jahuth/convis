import unittest
import shutil, tempfile
from os import path

class TestConvis(unittest.TestCase):
    def setUp(self):
        import convis
        self.convis = convis
        self.test_dir = tempfile.mkdtemp()
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    def test_retina(self):
        import convis
        import numpy as np
        ret = convis.retina.Retina()
        stimulus = np.zeros((2000,20,20))
        T,X,Y = np.meshgrid(np.arange(stimulus.shape[0]),
                            np.arange(stimulus.shape[1]),
                            np.arange(stimulus.shape[2]),indexing='ij')
        stimulus += 0.5*(1.0+np.sin(0.1 * T + 0.5*X + 0.3*Y))
        o = ret.run(stimulus, dt=200)
        self.assertEqual(len(o),2)

class TestConvisFilters(unittest.TestCase):
    def test_exponentials(self):
        import convis
        import numpy as np
        tau = 0.1
        resolution = convis.variables.ResolutionInfo(pixel_per_degree=10.0,filter_epsilon = 0.0001)
        f = convis.numerical_filters.exponential_filter_1d(tau,n=0,resolution=resolution)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Exponential filter is not normalized. %s"%str([np.abs(np.sum(f) - 1.0),0.1]))
        f = convis.numerical_filters.exponential_filter_1d(tau,n=4,resolution=resolution)
        self.assertTrue(np.abs(np.sum(f) - 1.0)<0.1,msg="Cascade exponential filter is not normalized. %s"%str([np.abs(np.sum(f) - 1.0),0.1]))
    def test_gaussians(self):
        import convis
        import numpy as np
        g = convis.numerical_filters.gauss_filter_2d(2.0,2.0)
        self.assertTrue(np.abs(np.sum(g) - 1.0)<0.1,msg="Gauss filter is not normalized.")




if __name__ == '__main__':
    unittest.main()
