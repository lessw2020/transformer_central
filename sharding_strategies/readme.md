#### Sharding Strategies for FSDP:

FSDP has 3 different sharding strategies which allow you to customize the tradeoff between memory vs communication, and thus with a single line of code, 
go from DDP -> Zero2 -> Full Shard. FSDP thus is becoming a universal training framework for models ranging from 100M - 1 Trillion+. 

In this video, you will learn how to modify the sharding strategy and understand the relative tradeoffs, and see the comparative growth in size of model trainable
on a fixed server, simply by adjusting the sharding strategy. 

Video:  https://www.loom.com/share/1f280618955c4f22a5a1ad6cbb41b28c

Notebook: FSDP_sharding_strategies.ipynb
