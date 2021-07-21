from setuptools import setup, find_packages
from glob import glob
from os.path import splitext
from os.path import basename


setup(
    name='optok_nmt',
    packages = ['optok_nmt'],
    version='0.1.0',
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    classifiers=[
        'Development Status :: 1 - Planning',
    ],
    install_requires=['numpy', 
                      'numba',
                      'scipy',
                      'pyyaml',
                      'sklearn',
                      'tqdm',
                      'pyyaml',
                      'torch']
)
