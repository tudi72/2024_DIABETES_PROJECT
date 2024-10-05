"""
Course Project 6: Imputation Model for Glucose Values Out of
Measuring Range for Continuous Glucose Monitors

Data Preprocessing

@brief Load data, remove 80th percentile, randomly mask
@date September 2024
@authors Di Fazio, Portmann, Zaharia
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants for the dictionary hashes
CBG:str = 'cbg'
TT:str = 'minutes_elapsed'
SI:str = 'start_indices'
EI:str = 'end_indices'
SL:str = 'segment_lengths'

class Preprocessing:
    def __init__(self, train_path:str, rand_seed:int = 42) -> None:
        np.random.seed(rand_seed)
        self._train_dict:dict = self._load_data(train_path)

        ### Splitting data into continous segments with no nans
        self._cont_segments_loc = self._get_cont_segments_loc(self._train_dict, CBG)        
        self._cont_segments = self._copy_cont_segments(self._cont_segments_loc, self._train_dict, CBG)

        self._nan_segments_loc = self._get_cont_nan_loc(self._train_dict, CBG)

        self._print_data_info()

    def _info(self, msg:str) -> None:
        """Info print utility to know which class prints to terminal."""
        print("\033[36m[Preprocessor]: ", msg, "\033[0m\n")

    def _load_data(self, path:str) -> dict:
        """Load data as dataframe and return dict with each patient"""

        patient_files = [patient for patient in os.listdir(path) if patient.endswith('.csv')]
        self._info(f"Loaded {len(patient_files)} files: {patient_files}")
        patient_dict = {}

        for index,patient in enumerate(patient_files):
            patient_dict[index] = pd.read_csv(os.path.join(path,patient))
            patient_dict[index][TT] = np.arange(0, len(patient_dict[index]) * 5, 5) 
        
        return patient_dict
    
    def _get_cont_segments_loc(self, dict:dict, col:str) -> dict[dict[np.ndarray]]:
        """Returns a dictionary of dicts where each data series / patient is indexed individually.
        For each index there is a field 'start_indices', 'end_indices' and 'segment_lengths'."""
        
        segments = {}
        for index in dict.keys():
            
            non_nan_indices = np.asarray(self._train_dict[index][col][self._train_dict[index][col].notna()].index) # Get indices for nan entries in cbg
            deriv = np.diff(non_nan_indices)
            end_indices = non_nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, non_nan_indices[-1])

            start_indices = non_nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, non_nan_indices[0])

            lengths = (end_indices-start_indices)+1
            segments[index] = {SI: start_indices, EI: end_indices, SL: lengths}

        return segments

    def _get_cont_nan_loc(self, dict:dict, col:str) -> dict[dict[np.ndarray]]:
        """Returns a dictionary of dicts where each data series / patient is indexed individually.
        For each index there is a field 'start_indices', 'end_indices' and 'segment_lengths'."""
        
        segments = {}
        for index in dict.keys():
            
            nan_indices = np.asarray(self._train_dict[index][col][self._train_dict[index][col].isna()].index) # Get indices for nan entries in cbg

            deriv = np.diff(nan_indices)
            end_indices = nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, nan_indices[-1])

            start_indices = nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, nan_indices[0])

            lengths = (end_indices-start_indices)+1
            segments[index] = {SI: start_indices, EI: end_indices, SL: lengths}
            # snippets[index] = np.asarray([start_indices, lengths])

        return segments
    
    def _copy_cont_segments(self, segments_loc:dict[dict[np.ndarray]], dict:dict, col:str):
        """Creates a new dict with dicts for each patient holding all segments without nans."""
        segments = {}
        for patient in segments_loc.keys():
            segments[patient] = []
            for index , loc in enumerate(segments_loc[patient][SI]):
                segments[patient].append(np.asarray(dict[patient][col][loc:segments_loc[patient][EI][index]]))

        return segments
    
    def _print_data_info(self) -> None:
        non_nan_total = 0
        nan_total = 0

        lens = []
        nan_lens = []

        for i in self._train_dict.keys():
            non_nan_total += len(self._cont_segments[i])
            nan_total += len(self._nan_segments_loc[i][SL])

            lens.extend(self._cont_segments_loc[i][SL])
            nan_lens.extend(self._nan_segments_loc[i][SL])

        lens = np.asarray(lens)
        nan_lens = np.asarray(nan_lens)

        lls = pd.DataFrame(lens, columns = ["not nans"])
        llls = pd.DataFrame(nan_lens, columns=["nans"])
        msg = f"train dict has {len(self._train_dict)} entries.\n" + f"Total of {non_nan_total} non nan sequences\n" + f"{lls.describe()}\n"
        msg += f"\nTotal of {nan_total} nan sequences\n" + f"{llls.describe()}\n"

        self._info(msg)


if __name__ == '__main__':
    TRAIN_PATH = 'Data/Ohio2020_processed/train'
    preproc = Preprocessing(TRAIN_PATH)

    train_segments = preproc._cont_segments
    plt.figure()
    n_plots = 5
    for i in range(5):
        plt.subplot(n_plots, 1, i+1)
        peaks, __ = find_peaks(train_segments[0][i], height=np.quantile(train_segments[0][i], 0.8), distance=50)
        plt.plot(train_segments[0][i])
        plt.scatter(peaks, train_segments[0][i][peaks])
        plt.grid('on')

    plt.tight_layout()
    plt.show()