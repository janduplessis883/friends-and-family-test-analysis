from setuptools import setup, find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name='friend-family-test',
    description="Analysis of Friends and Family Test responses received from individuals",
    packages=find_packages(),  # It will find all packages in your directory
    install_requires=requirements,  # This is the key line to install dependencies
    version='0.0.1'
)
