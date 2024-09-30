from setuptools import setup, find_packages


with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="workshop_model",
    version="0.0.1",
    author="Luca",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=requirements,
    packages=find_packages(),
    python_requires='>=3.6.2',
)
