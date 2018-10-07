import numpy as np
from scipy.signal import firwin, hilbert
import random
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import math

def generate_signal(length, freq_cutoff = [0.1, 0.15]):
    band = scipy.signal.firwin(34, freq_cutoff, fs=2, pass_zero = False)
    signal = np.random.normal(0, 1, length)
    filtered = np.convolve(signal, band, mode = "same")
    envelope = np.abs(scipy.signal.hilbert(filtered))
    return signal, filtered, envelope


def data_generator(X, Y, batch_size, lag_backward, lag_forward, shuffle = True, infinite = True):
    assert len(X)==len(Y) or len(Y)==0
    total_lag = lag_backward + lag_forward
    all_batches = math.ceil((X.shape[0] - total_lag)/batch_size)
    samples_in_last_batch = (X.shape[0] - total_lag) % batch_size
    batch = 0
    random_core = np.arange(lag_backward, X.shape[0]-lag_forward)
    while True:
        if shuffle: np.random.shuffle(random_core)
        for batch in range(all_batches):       
            batch_start = batch*batch_size
            batch_end = (batch+1)*batch_size
            if batch_end>=len(random_core): batch_end = None
            batch_samples = random_core[batch_start : batch_end]

            batch_x = np.array([X[i - lag_backward : i + lag_forward] for i in batch_samples])
            batch_x = np.swapaxes(batch_x,1,2)

            if len(Y) > 0:
                batch_y = Y[[batch_samples]] 
                yield (batch_x, batch_y)
            else:
                yield batch_x
        
        if not infinite:
            break


class SimulatedDataCreator:
    def __init__(self, number_of_input_signals, signals_length, target_lag):
        ## initialization basic params
        self.number_of_input_signals = number_of_input_signals
        self.signals_length = signals_length 
        self.mixing = number_of_input_signals
        self.broadband = 0.1
        self.low_filter_border = 0.05
        self.high_filter_border = 0.8
        self.target_lag = 10
        assert(self.low_filter_border>0 and self.high_filter_border+self.broadband<1), "Wrong Filters Borders"

        ## create random parameters
        self.mixing_matrix = np.random.uniform(0,1, (number_of_input_signals, number_of_input_signals))
        self.weight_matrix = np.random.uniform(0,1, (self.target_lag, number_of_input_signals))
        self.filters = np.random.uniform(self.low_filter_border,self.high_filter_border, self.number_of_input_signals)
        self.filters = np.array([(low_pass, low_pass + self.broadband) for low_pass in self.filters])

        ## create empty variables
        self.signals = []
        self.filtered_signals = []
        self.envelopes = []
        self.target = []

    def create(self,):
        for i in range(self.number_of_input_signals):
            signal, filtered, envelope = generate_signal(self.signals_length + self.target_lag - 1, self.filters[i])
            self.signals.append(signal)
            self.filtered_signals.append(filtered)
            self.envelopes.append(envelope)
        self.signals = np.array(self.signals).transpose()
        self.filtered_signals = np.array(self.filtered_signals).transpose()
        self.envelopes = np.array(self.envelopes).transpose()
        self.mixed_signals = np.matmul(self.signals, self.mixing_matrix)
        self.target = scipy.signal.convolve2d(self.envelopes, self.weight_matrix, mode='valid')
        self.signals = self.signals[self.target_lag-1:]
        self.filtered_signals = self.filtered_signals[self.target_lag-1:]
        self.envelopes = self.envelopes[self.target_lag-1:]
        self.mixed_signals = self.mixed_signals[self.target_lag-1:]
        assert(self.signals.shape ==
               self.filtered_signals.shape ==
               self.envelopes.shape ==
               (self.signals_length, self.number_of_input_signals)), "Dimenstions Mistmatch"
        assert(len(self.target)==self.signals_length), "Wrong target, expected size {size}, got {got}".format(size = self.signals_length, got = len(self.tagtet))
        
        
    def info(self,):
        print("### GENERAL INFO")
        print("Number of signals: ", self.number_of_input_signals)
        print("Signals Length: ", self.signals_length)
        print("Lag Length: ", self.target_lag)
        
        print()
        print("### FILRERS")
        print(pd.DataFrame(np.round(self.filters,2), columns = ["Low", "High"]))
        
        print()
        print("### MIXING MATRIX")
        print(pd.DataFrame(np.round(self.mixing_matrix,2),
                           index = ["W{}".format(i) for i in range(len(self.mixing_matrix))],
                           columns = ["Signal{}".format(i) for i in range(len(self.mixing_matrix))]))
        
        print()
        print("### WEIGHT MATRIX")
        print(pd.DataFrame(np.round(self.weight_matrix,2),
                           index = ["W{}".format(i) for i in range(len(self.weight_matrix))],
                           columns = ["Signal{}".format(i) for i in range(self.number_of_input_signals)]))       
        
    def demonstrate_random_signals(self, signals_to_demonstrate = 1):
        print("\n## SIGNALS DEMONSTRATION")
        samples = 300
        for i in range(signals_to_demonstrate):
            print()
            plt.figure(figsize=(16,3))
            plt.title("Signal {n} (low: {low}, high: {high})".format(n = i,
                                                                    low = round(self.filters[i][0],2),
                                                                    high = round(self.filters[i][1],2)))
            plt.plot(self.signals[:samples,i], label = "signal")
            plt.plot(self.filtered_signals[:samples,i], label = "filtered signal")
            plt.plot(self.envelopes[:samples,i], label = "envelope")
            plt.legend(loc = 1)
            plt.show()

            
    def demonstrate_targets(self, targets_to_demonstrate = 1):
        print("\n## SIGNALS DEMONSTRATION")
        samples = 300
        for i in range(targets_to_demonstrate):
            plt.figure(figsize=(16,3))
            plt.plot(self.envelopes[:samples,:])
            plt.legend(loc = 1)
            plt.show()
            plt.figure(figsize=(16,3))
            plt.plot(self.target[:samples,i], label = "target")
            plt.legend(loc = 1)
            plt.show()