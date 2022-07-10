### Maximizing your training speed with FSDP and gpu memory:

Conventional wisdom is that to maximize your training throughput, you should run your batch size up until you OOM, 
and then just slightly back off from there, and viola, optimal throughput.

This is not correct though as you need to optimize by ensuring you are not hitting cudaMalloc retries to get maximum speed!
This tutorial covers an example with tuning a 2B model and the improvements by avoiding retries (25% greater throughput vs conventional practice), as well as offers a utility you can add to your project to automatically monitor basic gpu info for optimizing. 



