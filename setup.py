from setuptools import setup, find_packages

setup(
    name='spectraljet',
    version='0.1.0',
    packages=find_packages(include=['spectraljet', 'spectraljet.*']),
    #url='https://github.com/HenryDayHall/jetTools',
    #long_description=open('README.md').read(),
    install_requires=[
        "awkward == 0.13.0",
        "numpy >= 1.16.1",
        "matplotlib >= 3.1.2",
        "scipy >= 1.3.3",
        "pygit2 >= 1.2.1"
    ]
)

