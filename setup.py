from setuptools import setup, Extension
import numpy
import os
from pipeline.config_parser import VERSION

os.environ["CC"] = "g++"

# define the extension module
cos_module_np = Extension('parse_pileup', sources=['pipeline/parse_pileup.cpp'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args = ["-std=c++0x"],)

setup(
    name="MetaCONNET",
    version=VERSION, 
    description="Neural network based tool for polishing metagenomics datasets", 
    author="Bingru Sun", 
    python_requires=">=3.8",
    scripts=["pipeline/correct.sh", "pipeline/mapping_tgs.sh", "pipeline/mapping.sh", 
    "pipeline/recover.sh", "pipeline/pipeline.sh", "pipeline/pipeline_tgs.sh"],              
    packages=["pipeline"],
    install_requires=[
        "pandas", 
        "keras",
        "tensorflow",
        "pandas"
        ], 
    package_data={'': ['training_model/meta_correction_ont_normalize.keras',
    'training_model/meta_recovery_ont_normalize.keras' ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": ["metaconnet=pipeline.main:run"]
    },
    ext_modules=[cos_module_np]
)
