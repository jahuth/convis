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
        ret.debug = True
        stimulus = np.zeros((2000,50,50))
        T,X,Y = np.meshgrid(np.arange(stimulus.shape[0]),
                            np.arange(stimulus.shape[1]),
                            np.arange(stimulus.shape[2]),indexing='ij')
        stimulus += 0.5*(1.0+np.sin(0.1 * T + 0.5*X + 0.3*Y))
        o = ret.run_in_chunks(stimulus, 200)
        self.assertEqual(len(o),2)
    def test_simple_models(self):
        import convis
        import numpy as np
        k1 = np.random.rand(30,5,5)
        k1 -= k1.mean()
        l = convis.filters.simple.K_3d_kernel_filter({'kernel': k1},name='L')
        n = convis.filters.simple.Nonlinearity(name='N',inputs = l )
        m1 = convis.make_model(n)
        target,err = m1.add_target(n.graph,error_func=lambda a,b: (a-b)**2)
        m1.add_output(convis.base.T.grad(err.mean(), l.parameters.kernel))
        m1.create_function()
        l.save_parameters_to_json(path.join(self.test_dir, 'test_model_m1_l.json'))
        l.load_parameters_from_json(path.join(self.test_dir, 'test_model_m1_l.json'))
        m1.save_parameters_to_json(path.join(self.test_dir, 'test_model_m1.json'))
        m1.load_parameters_from_json(path.join(self.test_dir, 'test_model_m1.json'))

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
