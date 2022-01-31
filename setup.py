from setuptools import setup, find_packages

requirements = []
with open("requirements.txt", 'r') as req_file:
    for line in req_file.readlines():
        line = line.strip()
        if line[0] == '#':
            continue
        if line[-1] == ',':
            line = line[:-1]
        requirements.append(line)

setup(
    name='spectraljet',
    version='0.1.0',
    packages=find_packages(include=['spectraljet', 'spectraljet.*']),
    #url='https://github.com/HenryDayHall/jetTools',
    #long_description=open('README.md').read(),
    install_requires=requirements
)

