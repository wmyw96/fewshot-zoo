# fewshot-zoo


## ProtoNet:

To reproduce the ProtoNet results for (n\_way = 5, n\_shot=5):

```
python train.py --exp_id protonet.exp_im1 --model protonet --gpu 0
```

The result is 65.13\% (valid), 66.52\% (test)

To reproduce the ProtoNet results for (n\_way = 5, n\_shot=1):

```
python train.py --exp_id protonet.exp_im2 --model protonet --gpu 0
```

The result is 48.12\% (valid), 48.91\% (test)
