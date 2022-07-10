### Maximizing your training speed with FSDP and gpu memory:

Conventional wisdom is that to maximize your training throughput, you should run your batch size up until you OOM, 
and then just slightly back off from there, and viola, optimal throughput.

This is not correct though as you need to optimize by ensuring you are not hitting cudaMalloc retries to get maximum speed!

This tutorial covers an example with tuning a 2B model and the improvements by avoiding retries (**25% greater throughput** vs conventional practice), 
as well as offers a utility you can add to your project to automatically monitor gpu info and retry counts for optimizing. 

Video:  https://www.loom.com/share/2dd1bb59468640df876578835603d0a7

Notebook: throughput_max.ipynb

The Memory_Maximizer class is also included in the attached gpu_memory.py, which automates the monitoring to implement the best practice in the tutorial. 



