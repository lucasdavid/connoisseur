try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='connoisseur',
    description='Machine Learning for art authorship recognition.',
    license='MIT License',
    long_description=open('README.md').read(),
    keywords=['machine-learning', 'paintings'],
    version='0.1',
    packages=['connoisseur'],
    scripts=[],

    author='Lucas David',
    author_email='ld492@drexel.edu',

    install_requires=open('docs/requirements-base.txt').readlines(),
    tests_require=open('docs/requirements-dev.txt').readlines(),
)
