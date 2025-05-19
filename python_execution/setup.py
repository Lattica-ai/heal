from setuptools import setup, find_packages

setup(
    name="lattica_heal_runtime",
    version="0.1.0",
    author="LatticaAI Inc.",
    author_email="support@lattica.ai",
    packages=find_packages(),
    install_requires=[
        "torch~=2.5.1",
        "numpy~=2.0.1"
    ],
    python_requires=">=3.7",
)
