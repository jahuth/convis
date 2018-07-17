import unittest
import shutil, tempfile
from os import path


class TestModels(unittest.TestCase):
    def setUp(self):
        import convis
        self.convis = convis
        self.test_dir = tempfile.mkdtemp()
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    def test_models(self):
        import convis
        import numpy as np
        for model in [convis.models.L((10,1,1)),
                      convis.models.LN(),
                      convis.models.LNLN(),
                      convis.models.LNCascade(3),
                      convis.models.Retina(),
                      convis.models.Retina(bipolar=False)]:
            inp = 100.0*np.random.rand(1000,20,20)
            o = model.run(inp,dt=200)
            #o.plot(mode='lines',label=model.__class__.__name__)
        #plt.legend()
        #plt.savefig('test_output/models.png')
    def test_filters(self):
        import convis
        import numpy as np
        for model in [convis.filters.Conv3d(),
                      convis.filters.Delay(10),
                      convis.filters.NLSquare(),
                      convis.filters.SmoothConv()]:
            inp = 100.0*np.random.rand(1000,20,20)
            o = model.run(inp,dt=200)


if __name__ == '__main__':
    unittest.main()
