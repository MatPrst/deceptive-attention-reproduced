# import unittest
#
# from utils import *
#
#
# class TestUtils(unittest.TestCase):
#
#     def test_bleu_score(self):
#         reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
#         candidate = ['this', 'is', 'a', 'test']
#
#         score = bleu_score(reference, candidate)
#         self.assertEqual(1.0, score)
#
#
# if __name__ == '__main__':
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
#     unittest.TextTestRunner(verbosity=2).run(suite)
