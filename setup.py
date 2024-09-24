from setuptools import setup, find_packages

setup(
    name='moving_mahalanobis',
    version='0.1',
    packages=find_packages(),  # Cela trouvera tous les packages Python dans ton projet
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn'
    ]
)
