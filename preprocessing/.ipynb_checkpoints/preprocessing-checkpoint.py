#!/usr/bin/env python
# coding: utf-8

# #### USE THIS FILE TO PREPROCESS THE GOOGLE MAPS DATA

# In[ ]:


import os
import pandas as pd
import glob
import shutil


# In[ ]:


class PreprocessData:
    # should contain all the functions as static methods
    @staticmethod
    def split_category(data_path: str):
        '''
        args: 
            data_path [string] -- path to image data saved with google screenshot code
        function:
            finds the satellite data and the map data and returns 2 lists with the individual absolute filepaths
        returns:
            2 lists with map_fnames, satellite_fnames
        interactions:
            Can be called by itself or by move_files function
        '''
        get_fnames = lambda name=None, path=data_path: glob.glob(os.path.join(os.path.abspath(path), name))
        return get_fnames('*-map-*.png'), get_fnames('*-sat-*.png')
    
    @staticmethod
    def move_files(data_path: str, sat_save_dir: str=None, map_save_dir: str=None):
        '''
        args:
            data_path [string] -- path to image data savd with goodle screenshot code
            sat_save_dir [string] -- path to directory to save the satellite RGB images
            map_save_dir [string] -- path to directory to save the map images
        function:
            takes the images from paths in the data_path and moves them to the specifies folders. 
            this is helpful if you want to move the images instead of just returning the filenames.
        returns:
            none
        interactions:
            None atm
        '''
        # get the absolute paths to map and satellite images
        map_fnames, sat_fnames = PreprocessData.split_category(data_path)

        # lambda function to move from soure to destination
        move = lambda source, destination: shutil.move(source, os.path.join(os.path.abspath(destination), os.path.basename(source)))

        # move map images
        for fname in map_fnames:
            move(fname, map_save_dir)

        # move sat fnames
        for fname in sat_fnames:
            move(fname, sat_save_dir)
            
    @staticmethod
    def create_data_csv(data_dir: str, csv_save_dir: str=None):
        '''
        args:
            data_dir [str] -- path to the /data folder
        function: 
            creates a csv, saves it in csv_save_dir and also returns the pandas DataFrame [sat_path, map_path]
        returns:
            pandas DataFrame 
        interactions: 
            None 
        '''
        data_frame = pd.DataFrame({'sat_path': glob.glob(os.path.join(data_dir, 'sat_image_data/*.png')),
                                   'map_path': glob.glob(os.path.join(data_dir, 'map_image_data/*.png'))})
        
        # save to path
        if csv_save_dir:
            data_frame.to_csv(csv_save_dir, columns=['sat_path', 'map_path'])
            
        return data_frame
    
    
    '''Need static method for removing street words from map'''
    
    '''Need static method for rempving building numbers '''
    
    '''Maybe need a method for converting into a segmentation map'''


# In[ ]:


# # path to data folder
# data_path = '../data/maps_data/'


# In[ ]:


# x,y = PreprocessData.split_category(data_path)


# In[ ]:


# PreprocessData.move_files(data_path, sat_save_dir='../data/sat_image_data/', map_save_dir='../data/map_image_data/')


# In[ ]:


# h = PreprocessData.create_data_csv('../data/')


# In[ ]:




