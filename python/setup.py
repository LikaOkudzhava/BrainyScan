from setuptools import setup, find_packages

setup(
    name='brainy',
    version='1.0',
    description='BrainyScan project supporting package',
    author='Viktar',
    author_email='myemail@example.com',
    packages=find_packages(where=''),
    install_requires=[
        'tensorflow',
        'keras',
        'numpy',
        'opencv-python',
        'tqdm',
        'scikit-learn',
        'seaborn',
    ]
)
