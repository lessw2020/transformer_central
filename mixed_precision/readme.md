### PyTorch FSDP Mixed Precision:

FSDP allows you to easily switch between various datatypes for your training via custom policies.
You can thus control the datatype for your parameters, your gradient communication and your buffers. 
Thus tutorial shows you how to do that as well as offers some best practices and a Bfloat16 checker module that 
will confirm both native GPU and network support for BFloat16. 

Video: https://www.loom.com/share/e308e10907884356ac263d269263e0bd

Notebook: mixed_precision.ipynb

Importable Code module: bfloat_verify.py

