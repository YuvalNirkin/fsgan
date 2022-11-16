import setuptools

setuptools.setup(
    name="fsgan",
    version="1.0.1",
    author="Dr. Yuval Nirkin",
    author_email="yuval.nirkin@gmail.com",
    description="FSGAN: Subject Agnostic Face Swapping and Reenactment",
    long_description_content_type="text/markdown",
    package_data={'': ['license.txt']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
