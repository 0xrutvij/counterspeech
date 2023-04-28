from setuptools import find_packages, setup

setup(
    name="counterspeech",
    version="0.4.8",
    description="Auto counterspeech generation for hatespeech",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "transformers",
        "datasets",
        "evaluate",
        "colorama",
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "spark-nlp>=4.4.0",
        "pyspark>=3.3.1",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
            "pylint",
        ]
    },
)
