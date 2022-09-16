With PyTorch Nightly version 914 and higher, there is now a new feature that can help free up additional GPU memory in some cases.

This is termed the 'rate limiter' and is used to ensure memory is not 'over buffered'.  Usage is simple, 
just set the new 'limit_all_gathers' during FSDP init to True to turn it on.

Here's a 4 minute video that shows you all the details:
https://www.loom.com/share/903330e463334f12912a31c333708960
