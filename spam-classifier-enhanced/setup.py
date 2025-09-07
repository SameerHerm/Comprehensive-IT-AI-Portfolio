from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spam-classifier-enhanced",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An enhanced spam classifier with multiple ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spam-classifier-enhanced",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
    ],
)
