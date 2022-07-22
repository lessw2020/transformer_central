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

6 - **Sharding Strategies for FSDP (video + notebook)**:

FSDP has 3 different sharding strategies which allow you to customize the tradeoff between memory vs communication, and thus with a single line of code, 
go from DDP -> Zero2 -> Full Shard. 
FSDP thus is becoming a universal training framework for models ranging from 100M - 1 Trillion+. 

In this tutorial, you will learn how to modify the FSDP sharding strategy, understand the relative tradeoffs, and see the comparative growth in size of model trainable on a fixed server simply by adjusting the sharding strategy. 

Video and notebook are in the folder [here](./sharding_strategies)

7 - **Mixed Precision with FSDP (video + notebook + importable module)**:

FSDP allows you to easily switch between various datatypes (Bfloat16, FP16, FP32) for your training via custom policies. 
You can thus control the datatype for your parameters, your gradient communication and your buffers. 
This tutorial shows you how to do that as well as offers some best practices and a Bfloat16 checker module that will confirm both native GPU and network support for BFloat16.

Video and notebook are [here](./mixed_precision)

8 - **Saving and Loading models with FSDP Local State Dict (distributed checkpoints)**:

FSDP has two methods for saving and loading models. Full State Dict saves and loads with the single file (.pt) concept. By contrast, Local State Dict saves to an exclusive directory, with potentially thousands of smaller files and a single .metadata file. The key benefit is local state dict allows model saving and loading for gigantic models where assembling it for single file saving and loading would exceed CPU memory.

This tutorial will show you how to work with local state / distributed checkpoints. The notebook has saving and loading functions you can directly leverage.

Video and notebook are [here](./saving_loading_models_local_state)

9 - **Fine Tuning Models with FSDP (video + notebook)**:

FSDP currently does not support layer level freezing for fine tuning (due to the sharding).  However, in this tutorial will discuss how to use Child Fine Tuning, which has been shown to outperform vanilla fine tuning on a variety of language tasks. 

Video and notebook are [here](./fine_tuning/readme.md)

10 - **End to End overview of FSDP in a working codebase (video)**:

This video tutorial does a 14 minute walkthrough of a codebase that is training a variety of models using FSDP.  The goal of this video is to show the overall features of FSDP within a codebase.  From there, you can dive into the detailed sub-tutorials on each specific topic of interest. 

Video is [here](./end_to_end)
