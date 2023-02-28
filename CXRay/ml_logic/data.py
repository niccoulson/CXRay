
import os
import pandas as pd
import numpy as np


if os.environ['DATA_SOURCE'] == 'local':
    chunk_df = get_xray_local()  # function to get local data

    return chunk_df

elif os.environ['DATA_SOURCE'] == 'cloud':
    chunk_df = get_xray_cloud() # Function to get cloud data

    return chunk_df

else: 
    raise ValueError(f'Value in .env {os.environ["DATA_SOURCE"]} is unknown')


