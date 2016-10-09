try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='connoisseur',
    description='Machine Learning for art authorship recognition.',
    long_description=open('README.md').read(),
    version='0.1',
    packages=['connoisseur'],
    scripts=[],
    author='Lucas David',
    author_email='ld492@drexel.edu',

    install_requires=['numpy', 'scipy', 'tensorflow', 'scikit-learn'],
    tests_require=open('requirements-dev.txt').readlines(),
)
