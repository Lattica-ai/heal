from setuptools import setup, find_packages

setup(
    name="lattica_heal_standalone_runtime",
    version="0.1.0",
    author="LatticaAI Inc.",
    author_email="support@lattica.ai",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    python_requires=">=3.7",
)