
## Loading and Saving model and optimizer checkpoints with FSDP and FULL_STATE_DICT

Loading and saving your model and optimizer checkpoints is an essential task during model training.  This tutorial covers how to do that in FSDP using FULL_STATE_DICT.  

FULL_STATE_DICT is an enum type within FSDP and refers to the fact that in this manner, all of the individual ranks data are combined onto rank0's cpu memory (making a complete or full state single dictionary), and aggregated into a single file that is then saved to disk via torch.save. 

This has the constraint that your full model params or optimizer params must be able to fit within cpu memory.  (think 20-30B size models and lower, but varies based on server).  
For models larger than this, we'll use LOCAL_STATE_DICT, which is a different tutorial and different process.  

If you are working with models that are likely to fit within your cpu memory, then we have both a video and notebook to help explain the process!

Details in our video here:
https://www.loom.com/share/ebb1d317d01a44b8aad54530a73e7bf1

and in the notebook:
load_save_full.ipynb

