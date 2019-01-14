
import doctest
import unittest

from tobacco_litigation import closing, corpus, dataset_creation, stats

def run_all_doctests():
    """
    Gather and runs all doctests

    :return:
    """

    test_suite = unittest.TestSuite()

    for module in [closing, corpus, dataset_creation, stats]:
        test_suite.addTest(doctest.DocTestSuite(module))

    results = unittest.TextTestRunner(verbosity=1).run(test_suite)
    return results


def run_doctests_travis():
    """
    Runs doctests for travis-ci and exits with code 1 if there were failures, 0 if there were none.

    :return:
    """

    results = run_all_doctests()
    if results.failures:
        exit(1)
    else:
        exit(0)


if __name__ == '__main__':

    run_all_doctests()
