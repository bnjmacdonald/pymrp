import os
from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='pymrp',
    version='0.0.1',
    packages=['pymrp', 'pymrp.mlm'],
    # include_package_data=True,
    license='MIT',
    description='A package for conducting multilevel regression and poststratification using Stan',
    long_description=README,
    url='https://github.com/bnjmacdonald/pymrp',
    author='Bobbie Macdonald',
    author_email='bnjmacdonald@gmail.com',
    install_requires = ['pystan'],  # TODO: add dependencies
    zip_safe=False
)
