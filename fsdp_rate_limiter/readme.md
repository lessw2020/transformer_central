With PyTorch Nightly version 914 and higher, there is now a new feature that can help free up additional GPU memory in some cases.

This is termed the 'rate limiter' and is used to ensure memory is not 'over buffered'.  Usage is simple, just set it at FSDP init to turn it on.

Video:
https://www.loom.com/share/903330e463334f12912a31c333708960
