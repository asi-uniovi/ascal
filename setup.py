from setuptools import setup, find_packages

setup(
    name="ascal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Jose Maria Lopez Lopez",
    author_email="chechu@uniovi.es",
    description="Simulate different H/HV and reactive/predictive autoscaling algorithms of containers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/asi-uniovi/ascal.git",  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
