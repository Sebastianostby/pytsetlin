from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="pytsetlin",
    version="1.1.0",
    author="Sebastian Østby",
    author_email="Sebastianostby@gmail.com",
    description="A low-code Pythonic implementation of a Coalesced Tsetlin Machine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sebastianostby/pytsetlin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)