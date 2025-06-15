from setuptools import setup, find_packages

setup(
    name='samplers_brothers',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    author='Manus AI',
    description='A Python library for time series sampling strategies based on clustering results.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/manus-ai/samplers_brothers', # Placeholder URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)


