from setuptools import setup

setup(
    name='TriosLibPython',
    version='2.2.1',
    packages=['trios', 'trios.legacy', 'trios.contrib', 'trios.contrib.nilc', 'trios.contrib.staffs',
              'trios.contrib.features', 'trios.contrib.kern_approx', 'trios.shortcuts', 'trios.classifiers',
              'trios.feature_extractors', 'trios.window_determination'],
    url='http://trioslib.github.io',
    license='OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
    author='André Lopes,Igor Montagner, Roberto Hirata Jr, Nina S. T. Hirata',
    author_email='igordsm+trios@gmail.com,andrelopes@anaccarati.com.br',
    description=''
)
