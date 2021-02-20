import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adacharge",
    version="0.1.0",
    author="Zachary Lee",
    author_email="zlee@caltech.edu",
    url="https://github.com/zach401/AdaCharge",
    description="MPC Algorithm for EV scheduling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"": ["LICENSE.txt", "THANKS.txt"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=["acnportal", "cvxpy", "numpy", "pytz"],
)
