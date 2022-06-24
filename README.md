# transformer_central
Various tutorials and transformers for FSDP research and learning.

# Lessons available:
1  - **Using the FSDP Transformer Wrapper (video + notebook)**

FSDP now has an express auto-wrapper for Transformer models.  This allows FSDP to create a 'model aware' sharding plan for how it breaks up the model across the GPU's and can result in some significant performance improvements for your training speed. 

Video and notebook are in the sub-folder [here](./transformer_wrapping_tutorial/readme.md)

2 - **Using FSDP's checkpoint activations (video + notebook)**

FSDP now has the ability to auto-insert checkpoints using a similar process as the Transformer wrapper where you designate your layer class.  Recommend watching the video 1 above first, before watching this video. 

Video and notebook are in the folder [here](./activation_checkpointing_tutorial/readme.md) 



