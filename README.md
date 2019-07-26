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

## DVE

mini-imagenet dataset pretrain

```
python train.py --pretype train --model dve --exp_id dve.im --gpu 1 --pretrain_dir saved_models/dve
```

train the model

```
python train.py --type train --model dve --exp_id dve.im --gpu 1 --pretrain_dir saved_models/dve
```
