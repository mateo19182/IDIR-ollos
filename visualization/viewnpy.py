import numpy as np
import pandas as pd

def view_npy_data(file_path, rows=10, cols=10, start_row=0, start_col=0):
    data = np.load(file_path)
    shape = data.shape
    print(f"Full data shape: {shape}")
    print(f"Data type: {data.dtype}")
    print(f"\nDisplaying {rows}x{cols} subset starting from position ({start_row}, {start_col}):\n")
    subset = data[start_row:start_row+rows, start_col:start_col+cols]
    df = pd.DataFrame(subset)
    print(df.to_string(index=False, header=False))


view_npy_data('dfv_01.npy', rows=1000, cols=1000, start_row=0, start_col=0)