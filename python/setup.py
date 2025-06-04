from setuptools import setup, find_packages

setup(
    name='brainy',
    version='1.1',
    description='BrainyScan project supporting package',
    author='Viktar',
    author_email='myemail@example.com',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'keras',
        'numpy',
        'opencv-python',
        'tqdm',
        'scikit-learn',
        'seaborn',
    ],
)
