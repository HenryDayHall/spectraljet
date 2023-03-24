import unittest

loader = unittest.TestLoader()
tests = loader.discover(start_dir='test', pattern='test_*.py')
test_runner = unittest.runner.TextTestRunner()
test_runner.run(tests)
