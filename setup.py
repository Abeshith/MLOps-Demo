from setuptools import setup, find_packages
from typing import List

def get_reqirements(file_path: str) -> List:
    """
    This function will return a list of requirements
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
            
    return requirements

setup(
    name='mlops_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=get_reqirements('requirements.txt'),
    description='A machine learning operations project',
    author='Abeshith',
    author_email="abeshith24@gmail.com",

)