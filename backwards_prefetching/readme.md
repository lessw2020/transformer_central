FSDP has multiple options for optimizing communication and computation overlap during the backward pass of training.

In this tutorial, we show the three current options, how to use them, and explain at a parameter level what the differences are. In some cases, switching to BACKWARD_PRE can improve your training speed up to 13% for a nominal memory increase of .59%.  Details below!

Video:  https://www.loom.com/share/21a465d89c6144f09d6a2e3d55824f1e

Notebook: backward_prefetch.ipynb



