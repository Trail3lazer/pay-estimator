import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

def run_parallel(df: pd.DataFrame, func: callable):
    cores = cpu_count()//2
    partitions = np.array_split(df, cores)
    p = Pool(cores)
    result = pd.concat(p.map(func, partitions))
    p.close()
    p.join()
    return result