from setuptools import setup, find_packages

base_requirements = open('requirements.txt').readlines()

setup(
    name='connoisseur',
    description='Machine Learning experiments on paintings',
    license='MIT License',
    keywords=['machine-learning', 'paintings'],
    version='0.1',
    packages=find_packages(),
    scripts=[],

    author='Lucas David',
    author_email='lucasolivdavid@gmail.com',

    setup_requires=base_requirements,
    install_requires=base_requirements,
)
