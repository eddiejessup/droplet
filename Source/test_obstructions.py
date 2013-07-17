import numpy as np
import unittest
import obstructions

class TestDroplet(unittest.TestCase):

	def setUp(self):
		L = 25.0
		dim = 3
		R = 10.0
		self.o = obstructions.Droplet(L, dim, R)

	def test_obstructed(self):
		R = np.array([0.1])
		r_inside = np.array([[0.0, 0.0]], dtype=np.float)
		r_outside = np.array([[self.o.R-R/2.0, 0.0]], dtype=np.float)
		self.assertFalse(self.o.is_obstructed(r_inside, R))
		self.assertTrue(self.o.is_obstructed(r_outside, R))

if __name__ == '__main__':
    unittest.main()