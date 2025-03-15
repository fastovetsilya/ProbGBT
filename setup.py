from setuptools import setup, find_packages

setup(
    name="prob-gbt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.22.4",
        "pandas>=2.2.2",
        "catboost>=1.2.3",
        "scikit-learn>=1.6.1",
        "tqdm>=4.67.1",
        "pygam>=0.8.0,<=0.9.1",
        "properscoring>=0.1",
        "matplotlib>=3.8.3",
        "scipy>=1.10.0",
        "geopandas>=1.0.1",
        "contextily>=1.6.2",
        "shapely>=2.0.3",
        "alphashape>=1.3.1",
    ],
    entry_points={
        'console_scripts': [
            'run-example=prob_gbt.example:main',
        ],
    },
    python_requires='>=3.9,<3.11',
    author="Ilya Fastovets",
    author_email="ilya.fastovets@gmail.com",
    description="Probabilistic Gradient Boosted Trees for uncertainty estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 