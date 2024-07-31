import json
import numpy as np
from pathlib import Path
import pickle
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime

#process_folder('/Users/aryamangupta/cric_pred/data/ipl_it20')#
json_files = sorted(
        Path('/Users/aryamangupta/cric_pred/data/ipl_it20').glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )

print(str(json_files[0]))

