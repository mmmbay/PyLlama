from setuptools import setup, find_packages

setup(
    name='pyllama',
    version='0.1',
    #packages=find_packages(where="pyllama"),  # Look inside "src"
    #package_dir={"": "pyllama"},  # Tell setuptools that packages are in "src"
    packages=find_packages(),
    install_requires=[],  # Add dependencies if needed
    include_package_data=True,
)
