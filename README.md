# transformer_central
Various tutorials and transformers for FSDP research and learning.

# Lessons available:
1  - **Using the FSDP Transformer Wrapper (video + notebook)**

FSDP now has an express auto-wrapper for Transformer models.  This allows FSDP to create a 'model aware' sharding plan for how it breaks up the model across the GPU's and can result in some significant performance improvements for your training speed. 

Video and notebook are in the sub-folder [here](./transformer_wrapping_tutorial)

2 - **Using FSDP's checkpoint activations (video + notebook)**

FSDP now has the ability to auto-insert checkpoints using a similar process as the Transformer wrapper where you designate your layer class.  Recommend watching the video 1 above first, before watching this video. 

Video and notebook are in the folder [here](./activation_checkpointing_tutorial) 

3 - **Loading and Saving Model and Optimizer checkpoints, with FULL_STATE_DICT (video + notebook)**

Saving and loading your training checkpoints is an essential task, and this tutorial covers how to do that with FSDP.  There are multiple state dict types within FSDP - this tutorial covers FULL_STATE_DICT, which is the typical use case within the constraint that the entire checkpoint (model or optimizer) is able to fit within your available cpu memory. 
For models that go beyond cpu memory (eg 20-30B+), we'll use distributed checkpoints via LOCAL_STATE_DICT which will be covered in a seperate tutorial.

Video and notebook are in the folder [here](./loading_saving_checkpoints_FULL_STATE_DICT)

4 - **Backwards Prefetching - optimize your training speed by increasing communication and computation overlap (video + notebook)**

FSDP has multiple options for optimizing communication and computation overlap during the backward pass of training.
In this tutorial, we show the three current options, how to use them, and explain at a parameter level what the differences are. 

Video and notebook are in the folder [here](./backwards_prefetching)

5 - **Maximizing your training speed with FSDP and gpu memory**:

Conventional wisdom is that to maximize your training throughput, you should run your batch size up until you OOM, 
and then just slightly back off from there, and viola, optimal throughput.

This is not correct though as you need to optimize by ensuring you are not hitting cudaMalloc retries to get maximum speed!

This tutorial covers an example with tuning a 2B model in FSDP, and the improvements by avoiding retries (**25% greater throughput** vs conventional practice), as well as offers a utility class (Memory_Maximizer) you can add to your project to automatically monitor gpu info and retry counts for optimizing. 

Video, notebook and utility file are in the folder [here](./throughput_max_gpu)
 








