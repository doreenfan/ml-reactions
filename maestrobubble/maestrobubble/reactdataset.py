
from torch.utils.data.dataset import Dataset
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import Video
from glob import glob
import torch
import warnings
import sys

import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
import gc

class ReactDataset(Dataset):

    def __init__(self, data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=False):
        #loading data
        #Load input and output data
        self.DEBUG_MODE = DEBUG_MODE
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.data_path = data_path

        self.input_files  = self.get_files(data_path, input_prefix)
        self.output_files = self.get_files(data_path, output_prefix)

        print("Loading Input Files...")
        self.input_data, self.input_break_file  = self.load_files(self.input_files, inputs=True)
        print("Loading Output Files...")
        self.output_data, self.output_break_file = self.load_files(self.output_files)

        #we want them to be the same length - so cut off data if we need to

        if (self.input_break_file is None) or (self.output_break_file is None):
            ind_in = len(self.input_files)
            ind_out = len(self.output_files)
        else:
            ind_in = self.input_files.index(self.input_break_file)
            ind_out = self.output_files.index(self.output_break_file)


        if ind_in <= ind_out:
            ind = ind_in
        else:
            ind = ind_out

        #cut excess.
        self.input_data = self.input_data[0:ind, :, :]
        self.output_data = self.output_data[0:ind, :, :]
        self.input_files = self.input_files[0:ind]
        self.output_files = self.output_files[0:ind]

        print("Loaded data successfully!")

    def get_files(self, data_path, prefix):
        data_files = glob(data_path + prefix)
        data_files = sorted(data_files)
        for data in data_files:
            if data[-7:] == 'endstep':
                data_files.remove(data)

        if self.DEBUG_MODE:
            data_files = data_files[:5]
        return data_files


    def load_files(self, file_list, inputs=False):

        break_file = None

        #Store data each row corresponds to data acros the grid of a different field.
        for j, file in enumerate(file_list):

            try:
                ds = yt.load(file)
                dt = ds.current_time.to_value()
                #Store data each row corresponds to data acros the grid of a different field.
                ymin = yt.YTArray(4.e7, 'cm')
                ymax = yt.YTArray(1.e8, 'cm')
                ad = ds.r[:, ymin:ymax]

                for i,field in enumerate(ds._field_list):
                    if i == 0:
                        data = np.zeros([len(ds._field_list), len(ad[field])])

                    data[i,:] = np.array(ad[field])
            except:
                pass


            if j == 0:
                #dt
                if inputs:
                    dt_tensor = dt*torch.ones([1,1,data.shape[1]])
                data = torch.from_numpy(data.reshape((1,data.shape[0],data.shape[1])))
                if inputs:
                    data = torch.cat((dt_tensor,data), dim=1)

                data_set = data
            else:

                try:
                    #dt
                    NUM_GRID_CELLS = data_set.shape[2]
                    if inputs:
                        dt_tensor = dt*torch.ones([1,1,data.shape[1]])

                    data = torch.from_numpy(data.reshape((1,data.shape[0],data.shape[1])))
                    #print(data.shape)
                    if inputs:
                        data = torch.cat((dt_tensor,data), dim=1)

                    #If we have more data - cut data
                    if data.shape[2] > NUM_GRID_CELLS:
                        data = data[:, : , :NUM_GRID_CELLS]
                    #We need to get more data.
                    elif data.shape[2] < NUM_GRID_CELLS:
                        #double size of cut
                        ad = ds.r[:, ymin/2:ymax*1.5]
                        for i,field in enumerate(ds._field_list):
                            if i == 0:
                                data = np.zeros([len(ds._field_list), len(ad[field])])

                            data[i,:] = np.array(ad[field])
                        data = torch.from_numpy(data.reshape((1,data.shape[0],data.shape[1])))
                        if inputs:
                            dt_tensor = dt*torch.ones([1,1,NUM_GRID_CELLS])
                        data = data[:, : , :NUM_GRID_CELLS]
                        if inputs:
                            data = torch.cat((dt_tensor,data), dim=1)


                    #z2 = data.reshape((1,data.shape[0],data.shape[1]))
                    #print(data.size())
                    data_set = torch.cat((data_set, data))
                    #torch.torch.stack([data_set,data], dim=0)
                except:
                    print('invalid data in file stopping here: ', file)
                    break_file = file
                    break

        return data_set, break_file


    def __getitem__(self, index):
        #indexing data dataset[0]

        #indexing goes across domain first, then to next data set.

        file_number = int(np.floor(index/self.input_data.shape[2]))
        cell_number = int(index%self.input_data.shape[2])

        #data
        X = self.input_data[file_number, :, cell_number]
        #labels
        Y = self.output_data[file_number, :, cell_number]

        return (X.float(),Y.float())

    def __len__(self):
        #Each cell contains a new training value.
        #The data is of the form [N1,N2,N3]
        #where N1 is the number of plot files we loaded
        #N2 is the inputs we will use in the model (state / thermo info)
        #N3 is the number of cells in that plotfile. Note this is 2D and we might
        #not want to store this as a 1d array in the future.

        #But for now we're simply just trying to learn a mapping between the input
        #thermo state and the output thermo state.
        return self.input_data.shape[2]* self.input_data.shape[0]


    def cut_data_set(self,N):
        #We have about 8gb of data and i can't train with that much when we're just testing.
        self.input_data = self.input_data[1:N,:,:]
        self.output_data = self.output_data[1:N,:,:]
