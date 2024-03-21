import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
format= '%(asctime)s : %(levelname)s %(message)s',
datefmt="%Y-%m-%d %A %H:%M:%S"
)
connet_logger = logging.getLogger("CONNET")
SAMTOOLS = "samtools"
VERSION = 1.0
MODEL1 = os.path.join(os.path.dirname(__file__), "training_model", "meta_correction_ont_normalize.keras")
MODEL2 = os.path.join(os.path.dirname(__file__), "training_model", "meta_recovery_ont_normalize.keras")
