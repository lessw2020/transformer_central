#### Fine Tuning Large Language models with FSDP

Currently FSDP does not support freezing layers that is often used in traditional fine tuning.  This may be added in the near future. 

However, this tutorial will show a potentially 'better way' than fine tuning with the whole model or layer freezing for large language models. 
This is known as ChildTuning from the paper by AliBaba and Peking University:
Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning (https://arxiv.org/abs/2109.05687)

The basic concept is to Bernoulli mask a subset of the parameters during the backward pass, and those selected receive a proportionate multiplier
to their gradients, in effect boosting a subset of the model for the specific task.  
This allows for both fine tuning while not losing generalization the models original pre-training work. 

We had good success with this for our 3B T5 Grammar checker workshop example and this tutorial show the main method to implement this for your tasks.

Video: 

Notebook: 




