from io import open
from setuptools import find_packages, setup

setup(
    name="checkers_ai",
    version="0.1",
    author="Chris Larson",
    author_email="chrismarclarson@gmail.com",
    description="source code for checkers.ai",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='checkers, draughts, reinforcement learning, deep neural networks, policy gradients, actor-critic, a2c',
    license='Apache',
    url="https://github.com/chrislarson1/checkers.ai",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=[
        'numpy',
        'requests',
        'tqdm',
        'regex',
        'aiohttp'
    ],
    python_requires='>=3.6.6',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
