#/bin/bash
conda env create -n metaconda -f env.yml
conda activate metaconda
pip install .
