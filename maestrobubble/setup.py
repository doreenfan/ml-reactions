from distutils.core import setup

setup(
    name='MaestroReactingBubble',
    version='0.1.0',
    author='Doreen Fan, Chris DeGrendele, Donald Willcox, Andy Nonaka',
    author_email='dfan@lbl.gov',
    packages=['maestrobubble'],
    license='LICENSE.txt',
    description='Machine Learning interface for learning a MAESTROeX reacting bubble.',
    long_description=open('README.md').read(),
    install_requires=[
        "matplotlib==3.4.1",
        "optuna==2.8.0",
        "pandas==1.1.3",
        "numpy==1.21.1",
        "yt==4.0.1",
        "torch==1.9.0",
        "ipython==7.27.0"
    ],
)
