import numpy as np

def fit_into_batch(data, batch_size:int=32):
    # mirroring data to make it can devide by batch_size
    max_samples_data = data.shape[0]
    if max_samples_data == batch_size:
        return data, max_samples_data
    padd_time = batch_size -  (max_samples_data-max_samples_data//batch_size*batch_size)
    fitted = data
    for _ in range(padd_time):
        fitted = np.concatenate((fitted, np.expand_dims(fitted[-1], axis=0)))
    return fitted, data.shape[0]

def extract_from_batch(data, extract_until:int):
    return data[:extract_until]
