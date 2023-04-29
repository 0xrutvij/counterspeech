from setuptools import find_packages, setup

setup(
    name="counterspeech",
    version="0.5.3",
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
