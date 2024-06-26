from setuptools import setup, find_packages

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(name="sksearchspace",
      version="2020.8.0.0.23.1",
      description="Defines search spaces for scikit-lean estimators",
      author="Thomas J. Fan",
      author_email="thomasjpfan@gmail.com",
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url="https://github.com/thomasjpfan/sksearchspace",
      packages=find_packages(include=["sksearchspace"]),
      include_package_data=True,
      python_requires='>=3.6',
      install_requires=["scikit-learn==1.5.0", "ConfigSpace>=0.4.13"],
      zip_safe=True,
      license='MIT')
