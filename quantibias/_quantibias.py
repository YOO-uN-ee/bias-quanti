import os
import warnings
import urllib3
import pickle

import polars as pl
import pandas as pd

from typing import List
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pl.MapWithoutReturnDtypeWarning)
# warnings.filterwarnings("ignore", category=urllib3.InsecureRequestWarning)
urllib3.disable_warnings()

from quantibias import data, converting, representation, linking

from quantibias._utils import (
    DefaultLogger,
    map_names,
)

logger = DefaultLogger()
logger.configure("INFO")

class FuseMine:
    def __init__(self,
                 model_name:str,
                 datasets:List[str],

                 verbose: bool=False,) -> None:
        
        

        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')