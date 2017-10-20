from setuptools import setup, find_packages

base_requirements = open('docs/requirements-base.txt').readlines()
dev_requirements = open('docs/requirements-dev.txt').readlines()

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

    setup_requires=base_requirements,
    install_requires=base_requirements,
    tests_require=dev_requirements,
)
