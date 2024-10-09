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
    def __init__(self, train_path:str, rand_seed:int = 42, debug:bool = True) -> None:
        np.random.seed(rand_seed)
        self._debug_flag:bool = debug

        self._patient_rawdata_dict:dict = self._load_data(train_path)

        """ Splitting data into continous segments with no nans"""
        self._data_segments_loc:dict  = self._get_cont_segments_loc(self._patient_rawdata_dict, CBG) # Gets location and length of non nan continous segments        
        self._data_segments:dict  = self._copy_cont_segments(self._data_segments_loc, self._patient_rawdata_dict, CBG) # Takes the actual CBG value in train dict at location from above

        # Does the same but for the segments which are nan
        self._nan_segments_loc:dict  = self._get_cont_nan_loc(self._patient_rawdata_dict, CBG)
        self._nan_segments:dict  = self._copy_cont_segments(self._nan_segments_loc, self._patient_rawdata_dict, CBG)

        # Before changing anything
        self._print_whole_data_info(plot_hist=False)
        self._patient_raw_metadata_dict = self._get_metadata()
        # self._print_patient_data_info(raw_or_processed='raw')

        """Pre Pre Processing"""
        self._remove_small_data_segments()
        self._resize_segments_uniform()
        self._control_data_segments:dict = self._data_segments.copy() # Create a copy of all data; later to be the train control

        """Pre Processing"""
        self._apply_quantile_cut() 
        self._apply_random_mask(min_non_nan=100)
        # Randome Masking
        # Collect pairs of images and store them       

        # Some info after processing
        self._patient_processed_metadata_dict = self._get_metadata()
        # self._print_patient_data_info(raw_or_processed='processed')


    def _info(self, msg:str) -> None:
        """Info print utility to know which class prints to terminal."""
        print("\033[36m\n[Preprocessor][INFO]: ", "\033[0m\n", msg)

    def _warn(self, msg:str) -> None:
        print("\033[93m\n[Preprocessor][WARNING]: ", msg, "\033[0m\n")

    def _debug(self, msg:str) -> None:
        if self._debug_flag:
            print("\033[35m\n[Preprocessor][DEBUG]: ", "\033[0m\n", msg)

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
            
            non_nan_indices = np.asarray(self._patient_rawdata_dict[index][col][self._patient_rawdata_dict[index][col].notna()].index) # Get indices for non nan entries in cbg
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
            
            nan_indices = np.asarray(self._patient_rawdata_dict[index][col][self._patient_rawdata_dict[index][col].isna()].index) # Get indices for nan entries in cbg

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
        """Returns a new dict with a list of np arrays for each patient. Each np array is a continous data segment."""
        segments = {}
        for patient in segments_loc.keys():
            segments[patient] = []
            for index , loc in enumerate(segments_loc[patient][SI]):
                segments[patient].append(np.asarray(dict[patient][col][loc:segments_loc[patient][EI][index]]))

        return segments
    
    def _print_whole_data_info(self, plot_hist:bool = True) -> None:
        """This function prints information about the amount and statistics of continous segments, both nan and non nan."""
        non_nan_total = 0
        nan_total = 0

        lens = []
        nan_lens = []

        for i in self._patient_rawdata_dict.keys():
            non_nan_total += len(self._data_segments[i])
            nan_total += len(self._nan_segments[i])

            lens.extend(self._data_segments_loc[i][SL])
            nan_lens.extend(self._nan_segments_loc[i][SL])

        lens = np.asarray(lens)
        nan_lens = np.asarray(nan_lens)

        if plot_hist:
            plt.subplot(2,1,1)
            plt.hist(lens, range=[0,1000], bins=100)
            plt.title("Hist of continous data segment lengths for all patients (in CBG)")
            plt.xticks(np.arange(0, 1000, step=50))
            plt.grid('on')

            plt.subplot(2,1,2)
            plt.hist(nan_lens, range=[0,1000], bins=100)
            plt.title("Hist of continous nan data segment lengths for all patients (in CBG)")
            plt.xticks(np.arange(0, 1000, step=50))
            plt.grid('on')
            plt.tight_layout()
            plt.show()

        lls = pd.DataFrame(lens, columns = ["not nans"])
        llls = pd.DataFrame(nan_lens, columns=["nans"])

        msg = f"train dict has {len(self._patient_rawdata_dict)} entries.\n" + f"Total of {non_nan_total} non nan sequences\n" + f"{lls.describe()}\n"
        msg += f"\nTotal of {nan_total} nan sequences\n" + f"{llls.describe()}\n"

        self._debug(msg)

    def _print_patient_data_info(self, raw_or_processed:str = 'raw') -> None:
        raw_bool = True if raw_or_processed == 'raw' else False

        if raw_bool:
            dict_in_question = self._patient_raw_metadata_dict
        else:
            dict_in_question = self._patient_processed_metadata_dict

        for key in dict_in_question.keys():
            dat = dict_in_question[key]['data']
            nan = dict_in_question[key]['nan']
            self._debug(f"+++ Patient_{key}_{'rawdata' if raw_bool else 'processed'}+++:\n {dat.describe()}\n\n {nan.describe()}\n\n")

    def _get_metadata(self) -> dict[pd.DataFrame]:
        """Similar to print info but this does it for each patient individually. The pd frames are stored for each patient key."""
        metadata = {}
        for key in self._patient_rawdata_dict.keys():
            # self._patient_metadata_dict[key] = {}
            metadata[key] = {}

            data_segs_lens = np.asarray([len(i) for i in self._data_segments[key]]) # Yes its a bit redundant
            nan_segs_lens = np.asarray([len(i) for i in self._nan_segments[key]])

            # self._patient_metadata_dict[key]['data'] = pd.DataFrame(data_segs, columns=[f"patient_{key}_data"])            
            # self._patient_metadata_dict[key]['nan'] = pd.DataFrame(nan_segs, columns=[f"patient_{key}_nan"])
            metadata[key]['data'] = pd.DataFrame(data_segs_lens, columns=[f"patient_{key}_data"]) # Create new dict so we can compare before and after processing           
            metadata[key]['nan'] = pd.DataFrame(nan_segs_lens, columns=[f"patient_{key}_nan"])

        return metadata

    def _remove_small_data_segments(self, min_non_nan:int = 100) -> None:
        """For now this removes from the continous data segments all the ones which are shorter than the longest nan segment."""
        for key, segments_list in self._data_segments.items():
            max_nan = self._patient_raw_metadata_dict[key]['nan'].max().item() # So the segments are at least as long as max nan seq.
            min_length = max_nan + min_non_nan # So it is by a given amount of samples larger at least.

            msg = f"MAX NAN patient_{key}: {max_nan}, min length defined to be: {min_length}\n"
            keep_segs = [seg for seg in segments_list if len(seg) >= min_length]
            msg+=f"Keeping longer data segments of lengths: {[len(seg) for seg in keep_segs]}\n"
            self._debug(msg)

            self._data_segments[key] = keep_segs
            # print(self._data_segments[key][0])

    def _resize_segments_uniform(self) -> None:
        """THis function resizes the segments sizes for each patient to the min which was larger than the largest nan segment. (Because for training all must be same) NOTE: THis is per patient not overall"""
        # loop over all keys
        for key in self._data_segments.keys():
            if not self._data_segments[key]:
                self._warn(f'In resize segments: No segments for key {key}. Omitting.')
                continue

            shortest_segment_len = np.asarray([len(i) for i in self._data_segments[key]]).min() # find the shortest segment per key
            self._debug(f"Cutting segments of patient_{key} down to length: {shortest_segment_len}")
           
            for i, seg in enumerate(self._data_segments[key]):
                # TODO: Possibly check if it is at least twice the size, then split it and append, to not waste data
                self._data_segments[key][i] = seg[:shortest_segment_len]# access the segment and slice from start to len
            
    def _apply_quantile_cut(self, nan_above_quantile:float = 0.8) -> None:
        for key in self._data_segments.keys():
            thresh = self._patient_rawdata_dict[key][CBG].quantile(nan_above_quantile) # NOTE we are using the complete dataset to define the quantile not each segment!
            msg = f'Patient_{key}: {nan_above_quantile} quantile threshold was {thresh}'
            self._debug(msg)

            for i,seg in enumerate(self._data_segments[key]):
                # print(type(seg), np.quantile(seg, 0.8))
                self._data_segments[key][i] = np.where(seg > thresh, np.nan, seg)

    def _apply_random_mask(self, min_non_nan:int = 100) -> None:
        """This function applies a mask to a random valid location in the segment. THe mask size is given by average nan segments length per patient."""
        # for loop over patient keys
        for key in self._data_segments.keys():

            if not self._data_segments[key]:
                self._warn(f"In apply random mask: No data for key: {key}. Omitting.")
                continue

            # Get the average nan segment length
            key_nan_lens = np.asarray([len(i) for i in self._nan_segments[key]])
            key_segment_len = len(self._data_segments[key][0])

            median_nan_length = np.floor(np.median(key_nan_lens))
            mean_nan_length = np.floor(np.mean(key_nan_lens))
            combined = (median_nan_length + mean_nan_length) // 2 # I couldnt decide whether mean or median is better so we do both.            
            self._debug(f"Patient_{key}: [mean, median, combined] nan segment length of [{mean_nan_length},{median_nan_length}, {combined}] and data segment length of {key_segment_len}")

            # Check if it is smaller than the segments in question
            if median_nan_length >= len(self._data_segments[key][0]):
                self._warn(f"In apply random mask: Patient_{key} mean nan segment length longer than segment! Omitting")
                continue

            # Get the valid indices for insertion

            # at random insert the nan mask

if __name__ == '__main__':
    TRAIN_PATH = 'Data/Ohio2020_processed/train'
    preproc = Preprocessing(TRAIN_PATH)

    train_segments = preproc._data_segments
    patient:int = 1
    plt.figure()
    n_plots = 5
    for i in range(5):
        plt.subplot(n_plots, 1, i+1)
        peaks, __ = find_peaks(train_segments[patient][i], height=np.quantile(train_segments[patient][i], 0.8), distance=50)
        plt.plot(train_segments[patient][i])
        plt.scatter(peaks, train_segments[patient][i][peaks])
        plt.grid('on')

    plt.tight_layout()
    plt.show()
