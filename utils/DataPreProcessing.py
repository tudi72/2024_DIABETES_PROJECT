"""
Course Project 6: Imputation Model for Glucose Values Out of
Measuring Range for Continuous Glucose Monitors

Data Preprocessing

@brief Load data, remove 80th percentile, randomly mask
@date September 2024
@authors Di Fazio, Portmann, Zaharia
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PreProcessor:
    def __init__(self, train_path:str, test_path:str, trhesh_quantile:float = 0.8, am_masks:int = 15, randSeed:int = 42) -> None:
        """
        PreProcessor class handles data loading, processing, and extracting NaN segments for training (TODO: and testing data).

        NOTE: Creating the instance automatically performs the processing.

        @param train_path: Path to the training data folder (grabs CSV format).
        @param test_path: Path to the testing data folder (grabs CSV format).
        @param thresh_quantile: Quantile threshold for censoring data above this percentile (default 0.8).
        @param am_masks: Number of random masking operations applied to each dataset for imputation (default 15).
        @param randSeed: Random seed for reproducibility (default 42).

        @return: None. Initializes the object with loaded data and processed results.
        
        Attributes:
        - train_dict (dict): Dictionary holding the loaded training data.
        - test_dict (dict): Dictionary holding the loaded testing data.
        - nan_segments (dict[dict[np.ndarray]]): Dictionary storing dictionary with the start indices and end indices and length of NaN segments for each time series.
        """
        np.random.seed(randSeed)
        self._train_dict:dict = self._load_data(train_path)
        self._test_dict:dict = self._load_data(test_path)

        self._process_data(thresh_quantile=trhesh_quantile, am_masks=am_masks)

        self._raw_data_segments:dict[dict[np.ndarray]] = self._extract_non_nan_segments(self._train_dict, 'cbg')
        self._raw_data_nan_segments:dict[dict[np.ndarray]] = self._extract_nan_segments(self._train_dict, 'cbg')

        self._all_nan_segments:dict[dict[np.ndarray]] = self._extract_nan_segments(self._train_dict, 'threshAndMasked')

        self._print_info_non_nan_segments(self._train_dict, 'cbg')

    def _load_data(self, path:str) -> dict:
        """Load data as dataframe and return dict with each patient"""

        patient_files = [patient for patient in os.listdir(path) if patient.endswith('.csv')]
        print(f"[loadData]: Loaded {len(patient_files)} files:\n ", patient_files)
        patient_dict = {}

        for index,patient in enumerate(patient_files):
            patient_dict[index] = pd.read_csv(os.path.join(path,patient))
            patient_dict[index]['minutes_elapsed'] = np.arange(0, len(patient_dict[index]) * 5, 5) 
        
        return patient_dict

    def _process_data(self, thresh_quantile:float, am_masks:int) -> None:
        """This function performs the thresholding and masking in sequence."""
        self._threshold_cbg(quantile=thresh_quantile)
        raw_data_nan_segments = self._extract_nan_segments(self._train_dict, 'cbg')
        self._random_masking(am_masks=am_masks, nan_segment_dict=raw_data_nan_segments)

    def _threshold_cbg(self, quantile:float = 0.8) -> None:
        """Creates new data column 'cbg_80th' from 'cbg' where values are below 'quantile' else np.nan"""

        for index in self._train_dict:
            patient_quantile_val = self._train_dict[index]['cbg'].quantile(quantile)
            self._train_dict[index]['cbg_80th'] = self._train_dict[index]['cbg'].where(
                self._train_dict[index]['cbg'] <= patient_quantile_val, np.nan
                )
    
    def _random_masking(self, am_masks:int, nan_segment_dict:dict[dict[np.ndarray]]) -> None:
        """
        Creates new data column 'threshAndMasked' from 'cbg_80th' where am_masks random windows are masked to np.nan
        The length of each mask window is chosen randomly from window_lengths.
        """
        all_window_lengths = []
        for i in range(len(nan_segment_dict)):
            all_window_lengths.extend(nan_segment_dict[i]["window_lengths"])
        
        for index in self._train_dict:
            self._train_dict[index]['threshAndMasked'] = self._train_dict[index]['cbg_80th']
            np.random.seed(index)
            rand_window_lengths = np.random.choice(all_window_lengths, size=am_masks, replace = False) # TODO: Move this out of the loop so we do not have to set a seed each time.

            for i, length in enumerate(rand_window_lengths):
                np.random.seed(i)
                max_index = len(self._train_dict[index]['cbg_80th'])-(length+1)
                rand_cbg_index = np.random.randint(0, max_index)
                self._train_dict[index].loc[rand_cbg_index:rand_cbg_index+length, 'threshAndMasked'] = np.nan

    def _print_info_nan_segments(self, dict:dict, col:str, plot:bool = True) -> np.ndarray:
        """Get insights into the missing data windows in training data.
        Returns an array with lengths of all nan sequences found."""

        nan_windows = self._extract_nan_segments(dict, col)
        all_lengths = []
        for i in range(len(nan_windows)):
            all_lengths.extend(nan_windows[i]["window_lengths"])

        all_lengths = np.asarray(all_lengths)
        df = pd.DataFrame(all_lengths, columns=[col])
 
        print('+'*8, 'INFO ON NAN WINDOWS', '+'*8)
        print(df[col].describe())

        # df['NaN windows'].hist(bins = 100)
        # plt.show()

        if plot:
            patient = 3
            nan_indices =  np.asarray(self._train_dict[patient]['cbg'][self._train_dict[patient]['cbg'].isna()].index) # Get indices for nan entries in cbg

            deriv = np.diff(nan_indices)
            end_indices = nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, nan_indices[-1])

            start_indices = nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, nan_indices[0])
            # Just as an example on what is being done
            
            plt.subplot(2,1,1)
            plt.plot(nan_indices)
            plt.title(f"NaN cbg indices on patient {patient}")
            plt.subplot(2,1,2)
            plt.plot(deriv)  
            plt.title("NaN cbg indices diff (scaling is off)")          
            plt.scatter(start_indices, np.zeros(len(start_indices)), color='green')
            plt.scatter(end_indices, np.zeros(len(end_indices)), color='red')
            plt.show()

        return all_lengths

    def _print_info_non_nan_segments(self, dict:dict, col:str, plot:bool = True) -> np.ndarray:
        """Get insights into the missing data windows in training data.
        Returns an array with lengths of all nan sequences found."""

        nan_windows = self._extract_non_nan_segments(dict, col)
        all_lengths = []
        for i in range(len(nan_windows)):
            all_lengths.extend(nan_windows[i]["window_lengths"])

        all_lengths = np.asarray(all_lengths)
        df = pd.DataFrame(all_lengths, columns=[col])
 
        print('+'*8, 'INFO ON NON NAN WINDOWS', '+'*8)
        print(df[col].describe())

    def _extract_nan_segments(self, dict:dict, col:str) -> dict[dict[np.ndarray]]:
        """Returns a dictionary of dicts where each data series / patient is indexed individually.
        For each index there is a field 'start_indices', 'end_indices' and 'window_lengths'."""
        
        segments = {}
        for index in dict.keys():
            
            nan_indices = np.asarray(self._train_dict[index][col][self._train_dict[index][col].isna()].index) # Get indices for nan entries in cbg

            deriv = np.diff(nan_indices)
            end_indices = nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, nan_indices[-1])

            start_indices = nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, nan_indices[0])

            lengths = (end_indices-start_indices)+1
            segments[index] = {"start_indices": start_indices, "end_indices": end_indices, "window_lengths": lengths}
            # snippets[index] = np.asarray([start_indices, lengths])

        return segments

    def _extract_non_nan_segments(self, dict:dict, col:str) -> dict[dict[np.ndarray]]:
        """Returns a dictionary of dicts where each data series / patient is indexed individually.
        For each index there is a field 'start_indices', 'end_indices' and 'window_lengths'."""
        
        segments = {}
        for index in dict.keys():
            
            non_nan_indices = np.asarray(self._train_dict[index][col][self._train_dict[index][col].notna()].index) # Get indices for nan entries in cbg
            deriv = np.diff(non_nan_indices)
            end_indices = non_nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, non_nan_indices[-1])

            start_indices = non_nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, non_nan_indices[0])

            lengths = (end_indices-start_indices)+1
            segments[index] = {"start_indices": start_indices, "end_indices": end_indices, "window_lengths": lengths}

        return segments

    def _remove_original_nan_segments(self) -> dict[dict[np.ndarray]]:
        """This function removes the nan segments we do actually not know. TODO: Is there a problem with overlapping segments?"""
        train_segments = {}
        # k = len(self._raw_data_nan_segments)
        # for i in range(k):
        #     orig_start = self._raw_data_nan_segments[i]["start_indices"]
        #     orig_end = self._raw_data_nan_segments[i]["end_indices"]
        #     orig_lengths = self._raw_data_nan_segments[i]["window_lengths"]

        #     processed_start = self._all_nan_segments[i]["start_indices"]
        #     processed_end = self._all_nan_segments[i]["end_indices"]
        #     processed_lengths = self._all_nan_segments[i]["window_lengths"]

        #     clean_start_indices = processed_start[~np.isin(processed_start, orig_start)]
        #     clean_end_indices = processed_end[~np.isin(processed_end, orig_end)]
        #     # Problem with lengths because they occur more than once!
        #     clean_lengths = processed_lengths[~np.isin(processed_start, orig_start)]
        #     train_segments[i] = {"start_indices": }

    def get_train_nan_segments(self) -> dict[dict[np.ndarray]]:
        """
        Get NaN segments of the cbg_80th and masked data series.
        Structure is: dict of dict with 3 np arrays.

        @example_usage
            nan_segments = preProc.getMissingSegments()

            patient_0_start_indices = nan_segments[0]["start_indices"]

            patient_0_end_indices = nan_segments[0]["end_indices"]      
                        
            patient_0_nan_lengths = nan_segments[0]["window_lengths"]
        """
        return self._all_nan_segments
    
    def get_train_non_nan_segments(self) -> dict[dict[np.ndarray]]:
        return self._raw_data_segments

    def get_train_dict(self) -> dict:
        """Get the dict of training data."""
        return self._train_dict
    
    def get_test_dict(self) -> dict:
        """Get the dict of test data."""
        return self._test_dict

if __name__ == '__main__':
    TRAIN_PATH = 'Data/Ohio2020_processed/train'
    TEST_PATH = 'Data/Ohio2020_processed/test'  

    ### Create instance and get then get the processed data
    preProc = PreProcessor(train_path=TRAIN_PATH, test_path=TEST_PATH)
    train_dict = preProc.get_train_dict()
    
    n = 6000
    
    nan_indicator = np.full(len(train_dict[0]['threshAndMasked']), np.nan)
    data_indicator = np.full(len(train_dict[0]['cbg']), np.nan)
    orig_nan_indicator = np.full(len(train_dict[0]['cbg']), np.nan)

    # VIsualizing correct extraction of nan windows
    nan_segments = preProc.get_train_nan_segments()

    # For the original data
    orig_nan = preProc._raw_data_nan_segments
    orig_data = preProc.get_train_non_nan_segments()

    patient_0_nan_indices = nan_segments[0]["start_indices"]
    patient_0_nan_lengths = nan_segments[0]["window_lengths"]

    patient_0_indices = orig_data[0]["start_indices"]
    patient_0_lengths = orig_data[0]["window_lengths"]

    patient_0_orig_indices = orig_nan[0]["start_indices"]
    patient_0_orig_lengths = orig_nan[0]["window_lengths"]

    for k, index in enumerate(patient_0_nan_indices):
        nan_indicator[index:index+patient_0_nan_lengths[k]] = 50

    for k, index in enumerate(patient_0_indices):
        data_indicator[index:index+patient_0_lengths[k]] = 50

    for k, index in enumerate(patient_0_orig_indices):
        orig_nan_indicator[index:index+patient_0_orig_lengths[k]] = 50

    low, high = 0,300
    plt.figure()
    
    plt.subplot(3,1,1)    
    plt.plot(train_dict[0]['minutes_elapsed'][0:n], train_dict[0]['cbg'][0:n])
    plt.scatter(train_dict[0]['minutes_elapsed'][0:n], orig_nan_indicator[0:n], color='red', marker='x')
    plt.scatter(train_dict[0]['minutes_elapsed'][0:n], data_indicator[0:n], color='green', marker='x')
    plt.ylim([low,high])
    plt.title("Raw Data")
    plt.grid('on')

    plt.subplot(3,1,2)
    plt.plot(train_dict[0]['minutes_elapsed'][0:n], train_dict[0]['cbg_80th'][0:n])
    plt.ylim([low,high])
    plt.title("80th quantile removed")
    plt.grid('on')

    plt.subplot(3,1,3)
    plt.plot(train_dict[0]['minutes_elapsed'][0:n], train_dict[0]['threshAndMasked'][0:n])
    plt.scatter(train_dict[0]['minutes_elapsed'][0:n], nan_indicator[0:n], color='red', marker='x')
    plt.ylim([low,high])
    plt.title("80th quantile removed and masked")  
    plt.grid('on')

    
    plt.tight_layout()
    plt.show()