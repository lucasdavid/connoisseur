from setuptools import setup, find_packages

setup(
    name='connoisseur',
    description='Machine Learning for art authorship recognition.',
    license='MIT License',
    long_description=open('README.md').read(),
    keywords=['machine-learning', 'paintings'],
    version='0.1',
    packages=find_packages(),
    scripts=[],

    author='Lucas David',
    author_email='lucasolivdavid@gmail.com',

    install_requires=open('docs/requirements-base.txt').readlines(),
    tests_require=open('docs/requirements-dev.txt').readlines(),
)
