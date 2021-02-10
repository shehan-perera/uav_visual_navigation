To use the preprocessing files you only need to execute the build_dataset.py or the build_dataset.ipynb file

build_dataset.py or build_dataset.ipynb will requrie 2 inputs:
  arg1 -- path to the folder that contains a folder called "maps_data" -- this folder is essential everything else will be created for you
          -- data 
          ----- maps_data
  arg2 -- a bool value, where if True then the dataset created and stored will tile the images squentially meaning you can put each tile back togther 
          in a logical way to create the original image. if False then the dataset is for training, these images are just tiles that are in random order and 
          putting the tiles back into a single image is not possible. 
          
the function to put the tiles back togther is WORK IN PROGRESS
