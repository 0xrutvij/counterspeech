from setuptools import find_packages, setup

setup(
    name="counterspeech",
    version="0.1",
    description="Auto counterspeech generation for hatespeech",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "transformers",
        "datasets",
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