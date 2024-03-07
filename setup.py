##Building application as a package


from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT=' -e .'
def get_requirements(file_path:str)->List[str]:
    """ This function will return a list of requirements from the file"""
    requiremtns = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.remove('\n',"") for req in requirements]
        
    
    if HYPHEN_E_DOT in requiremtns:
        requiremtns.remove(HYPHEN_E_DOT)
    
    return requirements

setup(

    name='mlproject',
    version='0.0.1',
    author='Shreyansh',
    author_email='shreyanshsing1998@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)