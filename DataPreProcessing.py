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
        raw_data_nan_window_Lengths = self.checkForNanWindows(self.train_dict, 'cbg', plot=False)  
        # plt.plot(self.train_dict[1]['minutes_elapsed'], self.train_dict[1]['cbg'])   # THis is the 738 samples missing, just checking if it was not a logic error.
        # plt.show()

        self.thresholdCBG()
        #self.window_lengths = self.checkForNanWindows(self.train_dict, 'thresholded', plot=False)

        self.randomMasking(am_masks=15, window_lengths=raw_data_nan_window_Lengths)
        self.window_lengths = self.checkForNanWindows(self.train_dict, 'threshAndMasked', plot=False)

        self.snippets = self.getImputationSnippets(self.train_dict, 'threshAndMasked')

    def loadData(self, path:str) -> dict:
        """Load data as dataframe and return dict with each patient"""

        patient_files = [patient for patient in os.listdir(path) if patient.endswith('.csv')]
        print(f"[loadData]: Loaded {len(patient_files)} files:\n ", patient_files)
        patient_dict = {}

        for index,patient in enumerate(patient_files):
            patient_dict[index] = pd.read_csv(os.path.join(path,patient))
            patient_dict[index]['minutes_elapsed'] = np.arange(0, len(patient_dict[index]) * 5, 5) 
        
        return patient_dict

    def getProcessedData(self, am_masks:int):
        pass

    def thresholdCBG(self, quantile:float = 0.8) -> None:
        """Creates new data column 'thresholded' from 'cbg' where values are below 'quantile' else np.nan"""

        for index in self.train_dict:
            patient_quantile_val = self.train_dict[index]['cbg'].quantile(quantile)
            self.train_dict[index]['thresholded'] = self.train_dict[index]['cbg'].where(
                self.train_dict[index]['cbg'] <= patient_quantile_val, np.nan
                )
    
    def randomMasking(self, am_masks:int, window_lengths:np.ndarray) -> None:
        """
        Creates new data column 'threshAndMasked' from 'thresholded' where am_masks random windows are masked to np.nan
        The length of each mask window is chosen randomly from window_lengths.
        """
        for index in self.train_dict:
            self.train_dict[index]['threshAndMasked'] = self.train_dict[index]['thresholded']
            np.random.seed(index)
            rand_window_lengths = np.random.choice(window_lengths, size=am_masks, replace = False)

            for i, length in enumerate(rand_window_lengths):
                np.random.seed(i)
                max_index = len(self.train_dict[index]['thresholded'])-(length+1)
                rand_cbg_index = np.random.randint(0, max_index)
                self.train_dict[index].loc[rand_cbg_index:rand_cbg_index+length, 'threshAndMasked'] = np.nan

    def checkForNanWindows(self, dict:dict, col:str, plot:bool = True) -> np.ndarray:
        """Get insights into the missing data windows in training data.
        Returns an array with lengths of all nan sequences found."""

        nan_windows = []
        for index in dict.keys():
            nan_indices = np.asarray(self.train_dict[index][col][self.train_dict[index][col].isna()].index) # Get indices for nan entries in cbg
            deriv = np.diff(nan_indices) # Derivative to highlight jumps in indices
            window_indices = np.argwhere(deriv!=1).ravel() # Where deriv = 1 are consecutive nan's, we want the jumps to find windows
            nan_windows.append(np.diff(window_indices)) # Derivative gives the length of the window (in samples) by difference of indices

        window_lengths = np.concatenate([np.array(window) for window in nan_windows])
        df = pd.DataFrame(window_lengths, columns=[col])
 
        print('+'*8, 'INFO ON NAN WINDOWS', '+'*8)
        print(df[col].describe())

        # df['NaN windows'].hist(bins = 100)
        # plt.show()

        if plot:
            # Just as an example on what is being done
            patient = 3
            train_nan = np.asarray(self.train_dict[patient]['cbg'][self.train_dict[patient]['cbg'].isna()].index)
            deriv = np.diff(train_nan)
            indices = np.argwhere(deriv!=1).ravel()
            plt.subplot(2,1,1)
            plt.plot(train_nan)
            plt.title(f"NaN cbg indices on patient {patient}")
            plt.subplot(2,1,2)
            plt.plot(deriv)  
            plt.title("NaN cbg indices diff")          
            plt.scatter(indices, np.zeros(len(indices)), color='red')
            plt.show()

        return window_lengths

    def getImputationSnippets(self, dict:dict, col:str, plot:bool = True) -> dict:
        """tbd"""

        nan_windows = []
        snippets = {}
        for index in dict.keys():
            
            nan_indices = np.asarray(self.train_dict[index][col][self.train_dict[index][col].isna()].index) # Get indices for nan entries in cbg

            deriv = np.diff(nan_indices)
            end_indices = nan_indices[np.where(deriv != 1)[0]]
            end_indices = np.append(end_indices, nan_indices[-1])

            start_indices = nan_indices[np.where(deriv != 1)[0]+1]
            start_indices = np.insert(start_indices, 0, nan_indices[0])

            lengths = (end_indices-start_indices)+1
            snippets[index] = np.asarray([start_indices, lengths])

        return snippets
           

if __name__ == '__main__':
    TRAIN_PATH = 'Data/Ohio2020_processed/train'
    TEST_PATH = 'Data/Ohio2020_processed/test'  

    ### Currently init does all the processing directly - this is testing phase still
    preProc = PreProcessor(train_path=TRAIN_PATH, test_path=TEST_PATH)
    
    n = 6000
    nan_indicator = np.full(len(preProc.train_dict[0]['threshAndMasked']), np.nan)
    for key, value in preProc.snippets.items():
        indices = value[0]
        lengths = value[1]
        for i, __ in enumerate(indices):
            nan_indicator[indices[i]:indices[i]+lengths[i]] = 50
        break # Only for the first one to check

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