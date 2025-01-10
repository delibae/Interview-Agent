from setuptools import find_packages, setup

setup(
    name="interviewagent",
    version="0.1.0",
    description="A package for managing interview data with Milvus and Jina embeddings.",
    author="delibae",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.25.1",
        "pymilvus>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
