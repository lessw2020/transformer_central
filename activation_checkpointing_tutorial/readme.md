## Using FSDP activation checkpointing

Activation checkpointing lets you avoid storing intermediate activations to free up memory, at the cost of re-doing compute.  
However, by freeing up memory, batch sizes can scale such that the total throughput for training time can be significantly increased and thus making activation checkpointing an important training tool for maximizing the speed of model training. 

FSDP checkpointing is shard aware and must be done after the FSDP init and sharding. 

Details in our video here:
https://www.loom.com/share/31749107033841959989aa8da45487c7

and in the notebook:
activation_checkpointing_tutorial.ipynb



