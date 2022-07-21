#### Loading and Saving Models with Local State Dict (distributed checkpointing)

FSDP has two methods for saving and loading models. Full State Dict saves and loads with the single file (.pt) concept.
By contrast, Local State Dict saves to an exclusive directory, with potentially thousands of smaller files and a single .metadata file. The key benefit is local state dict allows model saving and loading for gigantic models where assembling it for single file saving and loading would exceed CPU memory.

This tutorial will show you how to work with local state / distributed checkpoints. The notebook has saving and loading functions you can directly leverage.

Video: https://www.loom.com/share/be15a814a28145b9aebbaac683a44e3b

Notebook:

