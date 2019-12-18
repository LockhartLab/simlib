"""
test.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from simlib.tests.analysis import TestProtein

import sys
import unittest

# Define test cases
# TODO see if there's an easier way to look for these cases (without manually updating a list)
test_cases = [
    TestProtein,
]


# Function to run tests
def test():
    # Create a test suite
    suite = unittest.TestSuite()

    # Load all our test cases into the suite
    loader = unittest.TestLoader()
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))

    # Create test runner and run
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # If the test was not successful, exit with a code
    if not result.wasSuccessful():
        sys.exit(1)

