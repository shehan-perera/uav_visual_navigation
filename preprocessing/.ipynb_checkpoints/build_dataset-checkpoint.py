#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from preprocessing import PreprocessData
from sklearn.feature_extraction.image import extract_patches_2d
import argparse


# In[3]:


# function to read images
def read_image(path_to_img: str, array: bool=True):
    '''
    args:
        path_to_img [string] -- path to the image
    function:
        opens and returns the image as a numpy array
    returns:
        image array
    interactions:
        can be called by split_image    
    '''
    if array:
        return np.array(Image.open(path_to_img), dtype=np.uint8)
    else:
        return Image.open(path_to_img)


# In[4]:


def train_split_image(path_to_image: str, path_to_map: str, tile_shape: int=256, max_patches: int=30, combine: bool=True):
    '''
    args:
        path_to_img [string] -- path to the image
        tile_shape [int] -- the size of each of the tiles [tile_shape x tile_shape]
    function:
        tiles the image and returns a [tile_shape, tile_shape, num_tiles] array
        THIS FUNCTION IS USED TO CREATE IMAGE PATCHES FOR TRAINING, NOT TO BE USED FOR INFERENCE
        IF ANYTHING THIS CAN BE SEEN AS A PRETRAINING, BECAUSE WE ARE ONLY USING THE PATCHES, AND ONCE ITS PRETRAIND
            WE CAN THE ADD AN ADDITIONAL TERM WHERE WE NOT ONLY WANT THE PATCHES TO BE PREDICTED CORRECTLY WE WILL
            COMBINE THE PATCHES AND ADD A L1 LOSS TERM BETWEEN THE RECONSTUCTED MAP AND THE REAL MAP. 
    returns:
        returns a [tile_shape, tile_shape, num_tiles] array
    interactions:
        called by build_dataset()  
    '''
    # open image
    image = read_image(path_to_image, array=True)   
    map_img = read_image(path_to_map, array=True)
    
    # if combine is true combine the map and image data into single array
    # useful to have during but can be turned off at inference
    if combine:
        image = np.dstack((image, map_img))
        
    # resize the images so this function and inference_split_images will be the same
    resize_row = np.around(np.divide(image.shape[0], tile_shape)).astype(np.int) 
    resize_col = np.around(np.divide(image.shape[1], tile_shape)).astype(np.int) 
    resized_image = cv2.resize(image, (resize_col*tile_shape, resize_row*tile_shape), cv2.INTER_CUBIC)
                          
    # return the patches
    return extract_patches_2d(resized_image, patch_size=(tile_shape, tile_shape), max_patches=30, random_state=128)


# In[13]:


def save_tiles(tiles_array, image_path: str, map_path: str, tiles_dir_path: str):
    '''
    args: 
        tiles_array -- the tiles from train_split_image or from inference_split_image
        image_path -- the path used for the image 
        map_path -- the path used for the map
        tiles_dir_path -- the path to the directory that stores the tile data. dir should have 2 folders "image_tiles", "map_tiles"
    function:
        the image tiles and the map tiles will be saved into a directory using the same name from the image_path
    returns:
        none
    interactions:
        called by build_dataset()    
    '''
    
    # grab the iamge tiles
    image_tiles = tiles_array[:, :, :, 0:4]
    map_tiles = tiles_array[:, :, :, 4:]
    
    # get the name from the image path
    image_name = os.path.basename(image_path).split('.')[0]
    map_name = os.path.basename(map_path).split('.')[0]
    
    # save the image files
    for i in range(image_tiles.shape[0]):
        current_image_name = image_name + '_tile_' + str(i) + '.png'
        image_save_path = os.path.join(tiles_dir_path, 'image_tiles', current_image_name)
        cv2.imwrite(image_save_path, image_tiles[i, :, :, :])
        
    # save the map files
    for i in range(map_tiles.shape[0]):
        current_map_name = image_name + '_tile_' + str(i) + '.png'
        map_save_path = os.path.join(tiles_dir_path, 'map_tiles', current_map_name)
        cv2.imwrite(map_save_path, map_tiles[i, :, :, :])


# In[14]:


# for the inference split, the function should reshape the file while maintating the aspect ratio. 
def inference_split_image(path_to_image: str, path_to_map: str, tile_shape: int=256, combine: bool=True):
    '''
    args:
        path_to_img [string] -- path to the image
        tile_shape [int] -- the size of each of the tiles tile
    function:
        used for resizing the input image while maintaining the aspect ratio for inference
        this function is different from train_split_image b/c this one can be reconstructed back into the original image
        if combine is True, the tiled_data contains the image and the map tiles together to make 8 channels 
    return:
        tiled_data [num_tiles, height, width, 8] if combine is True else [num_tiles, height, width, 4]
    interactions:
        called by build_dataset()    
    '''
    # open image as array image
    image = read_image(path_to_image, array=True)
    map_img = read_image(path_to_map, array=True)
    
    # if combine is true combine the map and image data into single array
    # useful to have during but can be turned off at inference
    if combine:
        image = np.dstack((image, map_img))
            
    # resize keeping aspect ratio (maybe)
    resize_row = np.around(np.divide(image.shape[0], tile_shape)).astype(np.int) 
    resize_col = np.around(np.divide(image.shape[1], tile_shape)).astype(np.int) 
    resized_image = cv2.resize(image, (resize_col*tile_shape, resize_row*tile_shape), cv2.INTER_CUBIC)
        
    # image to array
    num_tiles = resize_row * resize_col
    tiled_data = np.zeros((num_tiles, tile_shape, tile_shape, image.shape[-1]), dtype=np.uint8)
    
    # split the image into tile_shape x tile_shape tiles
    for row in range(resize_row):
        for col in range(resize_col):
            count = row * col 
            start_row = row * tile_shape
            end_row = (row * tile_shape) + tile_shape
            start_col = col + tile_shape
            end_col = (col + tile_shape) + tile_shape
            tiled_data[count, :, :, :] = resized_image[start_row:end_row, start_col:end_col, :]
    
    #return tiles
    return tiled_data


# In[35]:


def build_dataset(path_to_data_folder: str, inference: bool=True):
    '''
    args:
        path_to_data_folder [string] -- the path to the folder that contains the folders "map_image_data" and "sat_image_data" and "maps_data" and "tiled_data"
    function:
        builds the data set
        first will use preprocessing.ProcessData to organize the data 
        next it will split the images into tiles and saves it into tiled_data/inference_tiles if inference is True else to tiled_data/training_files
    returns:
        None
    interactions:
        acts as main
    '''
    print('[info] -- started building dataset')
    
    # move the data from "maps_data" to "map_image_data" and "sat_image_data"
    original_data_path = os.path.join(path_to_data_folder, 'maps_data')
    map_image_data_path = os.path.join(path_to_data_folder, 'map_image_data')
    sat_image_data_path = os.path.join(path_to_data_folder, 'sat_image_data')
    
    print('[info] -- organizing data')
    # if map_image_data or sat_image_data are not empty dont preprocess
    if len(os.listdir(map_image_data_path)) != 0:
        os.mkdir(os.path.join(path_to_data_folder, 'tiled_data'))
        pass
    else:
        # if the folders dont exist create them
        os.mkdir(map_image_data_path)
        os.mkdir(sat_image_data_path)
        os.mkdir(os.path.join(path_to_data_folder, 'tiled_data'))
        PreprocessData.move_files(data_path=original_data_path, sat_save_dir=sat_image_data_path, map_save_dir=map_image_data_path)
    
    # get the csv file of the map and image data pairs 
    data_csv =  PreprocessData.create_data_csv(path_to_data_folder)
      
    if inference:
        print('[info] -- creating tiles for inference')
        # image save path
        save_dir = os.path.join(path_to_data_folder, 'tiled_data/inference_tiles')
        os.mkdir(save_dir)
        
        # create the directories to store the image data in
        os.mkdir(os.path.join(save_dir, 'image_tiles'))
        os.mkdir(os.path.join(save_dir, 'map_tiles'))
        
        for idx in tqdm(range(len(data_csv))):
            current_path_to_image = data_csv['sat_path'][idx]
            current_path_to_map = data_csv['map_path'][idx]
            current_tile = inference_split_image(current_path_to_image, current_path_to_map, tile_shape=256, combine=True)
            save_tiles(current_tile, current_path_to_image, current_path_to_map, save_dir)
    else:
        print('[info] -- creating tiles for training')
        # image save path
        save_dir = os.path.join(path_to_data_folder, 'tiled_data/training_tiles')
        os.mkdir(save_dir)
        
        # create the directories to store the image data in
        os.mkdir(os.path.join(save_dir, 'image_tiles'))
        os.mkdir(os.path.join(save_dir, 'map_tiles'))
        
        for idx in tqdm(range(len(data_csv))):
            current_path_to_image = data_csv['sat_path'][idx]
            current_path_to_map = data_csv['map_path'][idx]
            current_tile = train_split_image(current_path_to_image, current_path_to_map, tile_shape=256, max_patches=30, combine=True)
            save_tiles(current_tile, current_path_to_image, current_path_to_map, save_dir)
        


# In[39]:


# if running as notbook uncomment this
# build_dataset(path_to_data_folder='../../../data/', inference=False)


# In[ ]:


if __name__ == "__main__":

    # intialize parser
    parser = argparse.ArgumentParser(description='Information to build dataset')
    
    # arguments 
    parser.add_argument('data_folder', type=str, help='path to the folder that contains the data folder called "maps_data')
    parser.add_argument('inference', type=bool, help='true indicates budiling inference data and false indicates building trainig data')
    args = parser.parse_args()
    
    # build dataset
    build_dataset(args.data_folder, args.inference)
    
    print('[info] -- finished building dataset')


# In[ ]:





# In[ ]:


# working 

