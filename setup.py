from setuptools import setup, find_packages
import os


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        file_content = f.read()
    return file_content

def get_version():
    """Get version from the package without actually importing it."""
    init = read('speechaugs/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])

requirements = ["librosa>=0.6.1,<=0.8.0", "colorednoise>=1.1.1"]

setup(
    name="speechaugs",
    version=get_version(),
    author="Darya Vozhdaeva",
    author_email="daria-vozhdaeva@yandex.ru",
    description="Waveform augmentations",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/waveletdeboshir/speechaugs/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)