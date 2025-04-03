from setuptools import setup, find_packages


def get_requirements(filepath: str) -> list[str]:
    """
    Reads a requirements file and returns a list of requirements.

    Args:
        filepath (str): Path to the requirements file.

    Returns:
        list[str]: List of requirements.
    """
    try:
        with open(filepath, "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return []


setup(
    name="network-security-system",
    version="0.1.0",
    author="Sujeet Gund",
    author_email="sujeetgund@gmail.com",
    description="A network security system for detecting and preventing attacks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sujeetgund/network-security-system",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
