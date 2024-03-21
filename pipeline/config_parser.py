import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
format= '%(asctime)s : %(levelname)s %(message)s',
datefmt="%Y-%m-%d %A %H:%M:%S"
)
SAMTOOLS = "/AxBio_share/software/samtools-1.16/bin/samtools"
connet_logger = logging.getLogger("CONNET")
VERSION = 1.0
MODEL1 = os.path.join(os.path.dirname(__file__), "training_model", "meta_correction_ont_normalize.keras")
MODEL2 = os.path.join(os.path.dirname(__file__), "training_model", "meta_recovery_ont_normalize.keras")