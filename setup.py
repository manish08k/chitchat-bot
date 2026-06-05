from setuptools import setup, find_packages

setup(
    name="chitchat",
    version="1.0.0",
    packages=find_packages(exclude=["venv*", "data*", "__pycache__*"]),
)
