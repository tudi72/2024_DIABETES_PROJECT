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
    def __init__(self, train_path:str, test_path:str, randSeed:int = 42) -> None:
        """Create PreProc instance and laod data"""
        np.random.seed(randSeed)
        self.train_dict = self.loadData(train_path)
        self.test_dict = self.loadData(test_path)
        self.processData(am_masks=15)
        self.nan_windows = self.getNanSegments(self.train_dict, 'threshAndMasked')

    def loadData(self, path:str) -> dict:
        """Load data as dataframe and return dict with each patient"""

        patient_files = [patient for patient in os.listdir(path) if patient.endswith('.csv')]
        print(f"[loadData]: Loaded {len(patient_files)} files:\n ", patient_files)
        patient_dict = {}

        for index,patient in enumerate(patient_files):
            patient_dict[index] = pd.read_csv(os.path.join(path,patient))
            patient_dict[index]['minutes_elapsed'] = np.arange(0, len(patient_dict[index]) * 5, 5) 
        
        return patient_dict

    def processData(self, am_masks:int) -> None:
        """This function perofrms the thresholding and masking in sequence."""
        self.thresholdCBG(quantile=0.8)
        raw_data_nan_segments = self.getNanSegments(self.train_dict, 'cbg')
        self.randomMasking(am_masks=am_masks, nan_segment_dict=raw_data_nan_segments)

    def thresholdCBG(self, quantile:float = 0.8) -> None:
        """Creates new data column 'thresholded' from 'cbg' where values are below 'quantile' else np.nan"""

        for index in self.train_dict:
            patient_quantile_val = self.train_dict[index]['cbg'].quantile(quantile)
            self.train_dict[index]['thresholded'] = self.train_dict[index]['cbg'].where(
                self.train_dict[index]['cbg'] <= patient_quantile_val, np.nan
                )
    
    def randomMasking(self, am_masks:int, nan_segment_dict:dict[dict[np.ndarray]]) -> None:
        """
        Creates new data column 'threshAndMasked' from 'thresholded' where am_masks random windows are masked to np.nan
        The length of each mask window is chosen randomly from window_lengths.
        """
        all_window_lengths = []
        for i in range(len(nan_segment_dict)):
            all_window_lengths.extend(nan_segment_dict[i]["window_lengths"])
        
        for index in self.train_dict:
            self.train_dict[index]['threshAndMasked'] = self.train_dict[index]['thresholded']
            np.random.seed(index)
            rand_window_lengths = np.random.choice(all_window_lengths, size=am_masks, replace = False) # TODO: Move this out of the loop so we do not have to set a seed each time.

            for i, length in enumerate(rand_window_lengths):
                np.random.seed(i)
                max_index = len(self.train_dict[index]['thresholded'])-(length+1)
                rand_cbg_index = np.random.randint(0, max_index)
                self.train_dict[index].loc[rand_cbg_index:rand_cbg_index+length, 'threshAndMasked'] = np.nan

    def printInfoOnNanSegments(self, dict:dict, col:str, plot:bool = True) -> np.ndarray:
        """Get insights into the missing data windows in training data.
        Returns an array with lengths of all nan sequences found."""

        nan_windows = self.getNanSegments(dict, col)
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
            nan_indices =  np.asarray(self.train_dict[patient]['cbg'][self.train_dict[patient]['cbg'].isna()].index) # Get indices for nan entries in cbg

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

    def getNanSegments(self, dict:dict, col:str) -> dict[dict[np.ndarray]]:
        """Returns a dictionary of dicts where each data series / patient is indexed individually.
        For each index there is a field 'start_indices', 'end_indices' and 'window_lengths'."""
        
        segments = {}
        for index in dict.keys():
            
            nan_indices = np.asarray(self.train_dict[index][col][self.train_dict[index][col].isna()].index) # Get indices for nan entries in cbg

            deriv = np.diff(nan_indices)
            end_indices = nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, nan_indices[-1])

            start_indices = nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, nan_indices[0])

            lengths = (end_indices-start_indices)+1
            segments[index] = {"start_indices": start_indices, "end_indices": end_indices, "window_lengths": lengths}
            # snippets[index] = np.asarray([start_indices, lengths])

        return segments
           

if __name__ == '__main__':
    TRAIN_PATH = 'Data/Ohio2020_processed/train'
    TEST_PATH = 'Data/Ohio2020_processed/test'  

    ### Currently init does all the processing directly - this is testing phase still
    preProc = PreProcessor(train_path=TRAIN_PATH, test_path=TEST_PATH)
    
    n = 6000
    nan_indicator = np.full(len(preProc.train_dict[0]['threshAndMasked']), np.nan)

    # VIsualizing correct extraction of nan windows
    patient_0_nan_indices = preProc.nan_windows[0]["start_indices"]
    patient_0_nan_lengths = preProc.nan_windows[0]["window_lengths"]

    sum_windows = 0
    for i in range(len(preProc.nan_windows)):
        sum_windows += len(preProc.nan_windows[i]["window_lengths"])

    for k, index in enumerate(patient_0_nan_indices):
        nan_indicator[index:index+patient_0_nan_lengths[k]] = 50

    low, high = 0,300
    plt.figure()
    
    plt.subplot(3,1,1)    
    plt.plot(preProc.train_dict[0]['minutes_elapsed'][0:n], preProc.train_dict[0]['cbg'][0:n])
    plt.ylim([low,high])
    plt.title("Raw Data")
    plt.grid('on')

    plt.subplot(3,1,2)
    plt.plot(preProc.train_dict[0]['minutes_elapsed'][0:n], preProc.train_dict[0]['thresholded'][0:n])
    plt.ylim([low,high])
    plt.title("80th quantile removed")
    plt.grid('on')

    plt.subplot(3,1,3)
    plt.plot(preProc.train_dict[0]['minutes_elapsed'][0:n], preProc.train_dict[0]['threshAndMasked'][0:n])
    plt.scatter(preProc.train_dict[0]['minutes_elapsed'][0:n], nan_indicator[0:n], color='red', marker='x')
    plt.ylim([low,high])
    plt.title("80th quantile removed and masked")  
    plt.grid('on')

    
    plt.tight_layout()
    plt.show()