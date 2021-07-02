from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Model Trainer'
LONG_DESCRIPTION = 'Model Trainer Wrapper'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="soai", 
        version=VERSION,
        author="Prathaban",
        author_email="<prathaban@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'model trainer'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
        ]
)