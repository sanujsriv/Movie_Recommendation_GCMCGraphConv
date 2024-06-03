# Movie_Recommendation_GCMCGraphConv
This project was a part of a course - Movie_Recommendation_GCMCGraphConv

# Model Architecture, training samples & accuracy 
```
All rating pairs : 100000
	All train rating pairs : 75000
		Train rating pairs : 67500
		Valid rating pairs : 7500
	Test rating pairs  : 25000
Total user number = 72581, movie number = 39625
hello
Feature dim: 
user: (72581, 72581)
movie: (39625, 39625)
135000
150000
Train enc graph: 	#user:72581	#movie:39625	#pairs:67500
Train dec graph: 	#user:72581	#movie:39625	#pairs:67500
Valid enc graph: 	#user:72581	#movie:39625	#pairs:67500
Valid dec graph: 	#user:72581	#movie:39625	#pairs:7500
Test enc graph: 	#user:72581	#movie:39625	#pairs:75000
Test dec graph: 	#user:72581	#movie:39625	#pairs:25000
Loading data finished ...

Loading network finished ...

Start training ...
Total #Param of net: 56189420
Total Param Number: 56189420
Params:
	encoder.ufc.weight: torch.Size([75, 500]), 37500
	encoder.ufc.bias: torch.Size([75]), 75
	encoder.ifc.weight: torch.Size([75, 500]), 37500
	encoder.ifc.bias: torch.Size([75]), 75
	encoder.conv.mods.1.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-1.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.2.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-2.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.3.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-3.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.4.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-4.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.5.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-5.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.6.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-6.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.7.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-7.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.8.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-8.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.9.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-9.weight: torch.Size([39625, 50]), 1981250
	encoder.conv.mods.10.weight: torch.Size([72581, 50]), 3629050
	encoder.conv.mods.rev-10.weight: torch.Size([39625, 50]), 1981250
	decoder.Ps.0: torch.Size([75, 75]), 5625
	decoder.Ps.1: torch.Size([75, 75]), 5625
	decoder.combine_basis.weight: torch.Size([10, 2]), 20
Net(
  (_act): LeakyReLU(negative_slope=0.1)
  (encoder): GCMCLayer(
    (ufc): Linear(in_features=500, out_features=75, bias=True)
    (ifc): Linear(in_features=500, out_features=75, bias=True)
    (dropout): Dropout(p=0.7, inplace=False)
    (W_r): None
    (conv): HeteroGraphConv(
      (mods): ModuleDict(
        (1): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-1): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (2): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-2): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (3): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-3): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (4): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-4): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (5): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-5): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (6): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-6): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (7): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-7): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (8): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-8): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (9): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-9): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (10): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
        (rev-10): GCMCGraphConv(
          (dropout): Dropout(p=0.7, inplace=False)
        )
      )
    )
    (agg_act): LeakyReLU(negative_slope=0.1)
  )
  (decoder): BiDecoder(
    (dropout): Dropout(p=0.0, inplace=False)
    (Ps): ParameterList(
        (0): Parameter containing: [torch.cuda.FloatTensor of size 75x75 (GPU 0)]
        (1): Parameter containing: [torch.cuda.FloatTensor of size 75x75 (GPU 0)]
    )
    (combine_basis): Linear(in_features=2, out_features=10, bias=False)
  )
)
/home/grad16/sakumar/miniconda3/envs/cuda/lib/python3.8/site-packages/numpy/lib/function_base.py:380: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis)
/home/grad16/sakumar/miniconda3/envs/cuda/lib/python3.8/site-packages/numpy/core/_methods.py:188: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Iter=1, loss=2.3031, rmse=11.0005, time=nan,	Val RMSE=3.2827, Test RMSE=3.3088
Iter=2, loss=2.2993, rmse=10.9264, time=nan,	Val RMSE=3.2646, Test RMSE=3.2918
Iter=3, loss=2.2942, rmse=10.8063, time=nan,	Val RMSE=3.2337, Test RMSE=3.2630
Iter=4, loss=2.2870, rmse=10.5919, time=0.0566,	Val RMSE=3.1881, Test RMSE=3.2209
Iter=5, loss=2.2779, rmse=10.2618, time=0.0566,	Val RMSE=3.1403, Test RMSE=3.1773
Iter=6, loss=2.2691, rmse=9.9109, time=0.0565,	Val RMSE=3.1241, Test RMSE=3.1626
Iter=7, loss=2.2581, rmse=9.7584, time=0.0566,	Val RMSE=3.1202, Test RMSE=3.1586
Iter=8, loss=2.2446, rmse=9.6098, time=0.0566,	Val RMSE=3.1179, Test RMSE=3.1554
Iter=9, loss=2.2302, rmse=9.5010, time=0.0566,	Val RMSE=3.1068, Test RMSE=3.1435
Iter=10, loss=2.2162, rmse=9.3043, time=0.0566,	Val RMSE=3.0857, Test RMSE=3.1211
Iter=11, loss=2.2023, rmse=9.0491, time=0.0566,	Val RMSE=3.0634, Test RMSE=3.0951
Iter=12, loss=2.1887, rmse=8.8020, time=0.0566,	Val RMSE=3.0473, Test RMSE=3.0736
Iter=13, loss=2.1759, rmse=8.5482, time=0.0566,	Val RMSE=3.0312, Test RMSE=3.0548
Iter=14, loss=2.1642, rmse=8.3480, time=0.0566,	Val RMSE=3.0133, Test RMSE=3.0373
Iter=15, loss=2.1525, rmse=8.0896, time=0.0566,	Val RMSE=3.0047, Test RMSE=3.0298
Iter=16, loss=2.1411, rmse=7.8811, time=0.0566,	Val RMSE=3.0078
Iter=17, loss=2.1290, rmse=7.7657, time=0.0566,	Val RMSE=3.0081
Iter=18, loss=2.1166, rmse=7.6238, time=0.0566,	Val RMSE=2.9945, Test RMSE=3.0237
Iter=19, loss=2.1042, rmse=7.4401, time=0.0566,	Val RMSE=2.9737, Test RMSE=3.0063
Iter=20, loss=2.0919, rmse=7.2054, time=0.0566,	Val RMSE=2.9561, Test RMSE=2.9912
Iter=21, loss=2.0798, rmse=6.9165, time=0.0566,	Val RMSE=2.9471, Test RMSE=2.9829
Iter=22, loss=2.0679, rmse=6.7888, time=0.0566,	Val RMSE=2.9387, Test RMSE=2.9749
Iter=23, loss=2.0562, rmse=6.5744, time=0.0566,	Val RMSE=2.9320, Test RMSE=2.9687
Iter=24, loss=2.0444, rmse=6.4754, time=0.0566,	Val RMSE=2.9258, Test RMSE=2.9634
Iter=25, loss=2.0329, rmse=6.3492, time=0.0566,	Val RMSE=2.9203, Test RMSE=2.9596
Iter=26, loss=2.0219, rmse=6.1595, time=0.0566,	Val RMSE=2.9178, Test RMSE=2.9590
Iter=27, loss=2.0120, rmse=6.2184, time=0.0566,	Val RMSE=2.9156, Test RMSE=2.9575
Iter=28, loss=2.0022, rmse=6.1410, time=0.0566,	Val RMSE=2.9134, Test RMSE=2.9550
Iter=29, loss=1.9923, rmse=6.0070, time=0.0566,	Val RMSE=2.9131, Test RMSE=2.9540
Iter=30, loss=1.9821, rmse=5.9910, time=0.0566,	Val RMSE=2.9113, Test RMSE=2.9524
Iter=31, loss=1.9722, rmse=5.8749, time=0.0566,	Val RMSE=2.9063, Test RMSE=2.9490
Iter=32, loss=1.9627, rmse=5.8653, time=0.0565,	Val RMSE=2.9036, Test RMSE=2.9477
Iter=33, loss=1.9534, rmse=5.9290, time=0.0566,	Val RMSE=2.9032, Test RMSE=2.9481
Iter=34, loss=1.9445, rmse=5.9397, time=0.0566,	Val RMSE=2.9031, Test RMSE=2.9494
Iter=35, loss=1.9357, rmse=5.8667, time=0.0565,	Val RMSE=2.9032
Iter=36, loss=1.9267, rmse=5.8271, time=0.0565,	Val RMSE=2.9047
Iter=37, loss=1.9181, rmse=5.8652, time=0.0565,	Val RMSE=2.9074
Iter=38, loss=1.9096, rmse=5.8797, time=0.0565,	Val RMSE=2.9101
Iter=39, loss=1.9011, rmse=5.7638, time=0.0565,	Val RMSE=2.9136
Iter=40, loss=1.8931, rmse=5.8623, time=0.0565,	Val RMSE=2.9162
Iter=41, loss=1.8853, rmse=5.8386, time=0.0565,	Val RMSE=2.9175
Iter=42, loss=1.8778, rmse=5.7780, time=0.0565,	Val RMSE=2.9172
Iter=43, loss=1.8705, rmse=5.7948, time=0.0565,	Val RMSE=2.9158
Iter=44, loss=1.8634, rmse=5.7467, time=0.0565,	Val RMSE=2.9161
Iter=45, loss=1.8564, rmse=5.6917, time=0.0565,	Val RMSE=2.9152
Iter=46, loss=1.8497, rmse=5.7387, time=0.0565,	Val RMSE=2.9155
Iter=47, loss=1.8429, rmse=5.6723, time=0.0565,	Val RMSE=2.9184
Iter=48, loss=1.8363, rmse=5.7201, time=0.0565,	Val RMSE=2.9216
Iter=49, loss=1.8299, rmse=5.6990, time=0.0565,	Val RMSE=2.9241
Iter=50, loss=1.8238, rmse=5.6469, time=0.0565,	Val RMSE=2.9265
Iter=51, loss=1.8178, rmse=5.7572, time=0.0565,	Val RMSE=2.9269
Iter=52, loss=1.8118, rmse=5.6556, time=0.0565,	Val RMSE=2.9264
Iter=53, loss=1.8061, rmse=5.7071, time=0.0565,	Val RMSE=2.9273
Iter=54, loss=1.8005, rmse=5.7101, time=0.0565,	Val RMSE=2.9282
Iter=55, loss=1.7951, rmse=5.6999, time=0.0565,	Val RMSE=2.9271
Iter=56, loss=1.7897, rmse=5.6485, time=0.0565,	Val RMSE=2.9265
Iter=57, loss=1.7847, rmse=5.6616, time=0.0565,	Val RMSE=2.9263
Iter=58, loss=1.7796, rmse=5.6301, time=0.0565,	Val RMSE=2.9280
Iter=59, loss=1.7747, rmse=5.6672, time=0.0565,	Val RMSE=2.9304
Iter=60, loss=1.7698, rmse=5.5944, time=0.0565,	Val RMSE=2.9311
Iter=61, loss=1.7650, rmse=5.6055, time=0.0565,	Val RMSE=2.9309
Iter=62, loss=1.7604, rmse=5.6368, time=0.0565,	Val RMSE=2.9305
Iter=63, loss=1.7558, rmse=5.6308, time=0.0565,	Val RMSE=2.9302
Iter=64, loss=1.7515, rmse=5.5908, time=0.0565,	Val RMSE=2.9300
Iter=65, loss=1.7472, rmse=5.6302, time=0.0565,	Val RMSE=2.9321
Iter=66, loss=1.7430, rmse=5.5990, time=0.0565,	Val RMSE=2.9344
Iter=67, loss=1.7389, rmse=5.5608, time=0.0565,	Val RMSE=2.9354
Iter=68, loss=1.7349, rmse=5.5884, time=0.0565,	Val RMSE=2.9353
Iter=69, loss=1.7310, rmse=5.6026, time=0.0565,	Val RMSE=2.9357
Iter=70, loss=1.7271, rmse=5.5421, time=0.0565,	Val RMSE=2.9362
Iter=71, loss=1.7234, rmse=5.5936, time=0.0565,	Val RMSE=2.9375
Iter=72, loss=1.7196, rmse=5.5843, time=0.0565,	Val RMSE=2.9383
Iter=73, loss=1.7161, rmse=5.6019, time=0.0565,	Val RMSE=2.9368
Iter=74, loss=1.7126, rmse=5.6086, time=0.0565,	Val RMSE=2.9354
Iter=75, loss=1.7091, rmse=5.5447, time=0.0565,	Val RMSE=2.9357
Iter=76, loss=1.7057, rmse=5.5271, time=0.0565,	Val RMSE=2.9386
Iter=77, loss=1.7024, rmse=5.6578, time=0.0565,	Val RMSE=2.9419
Iter=78, loss=1.6990, rmse=5.5091, time=0.0565,	Val RMSE=2.9453
Iter=79, loss=1.6957, rmse=5.5061, time=0.0565,	Val RMSE=2.9468
Iter=80, loss=1.6925, rmse=5.5809, time=0.0565,	Val RMSE=2.9457
Iter=81, loss=1.6894, rmse=5.5270, time=0.0565,	Val RMSE=2.9456
Iter=82, loss=1.6862, rmse=5.4457, time=0.0565,	Val RMSE=2.9468
Iter=83, loss=1.6832, rmse=5.5465, time=0.0565,	Val RMSE=2.9478
Iter=84, loss=1.6804, rmse=5.5346, time=0.0565,	Val RMSE=2.9486
Iter=85, loss=1.6775, rmse=5.5943, time=0.0565,	Val RMSE=2.9496
Iter=86, loss=1.6748, rmse=5.4810, time=0.0565,	Val RMSE=2.9487
Iter=87, loss=1.6720, rmse=5.5236, time=0.0565,	Val RMSE=2.9471
Iter=88, loss=1.6693, rmse=5.5225, time=0.0565,	Val RMSE=2.9469
Iter=89, loss=1.6666, rmse=5.4969, time=0.0565,	Val RMSE=2.9476
Iter=90, loss=1.6640, rmse=5.5710, time=0.0565,	Val RMSE=2.9481
Iter=91, loss=1.6614, rmse=5.4971, time=0.0565,	Val RMSE=2.9484
Iter=92, loss=1.6588, rmse=5.5164, time=0.0565,	Val RMSE=2.9486
Iter=93, loss=1.6563, rmse=5.6178, time=0.0565,	Val RMSE=2.9487
Iter=94, loss=1.6539, rmse=5.5396, time=0.0565,	Val RMSE=2.9488
Iter=95, loss=1.6515, rmse=5.5078, time=0.0565,	Val RMSE=2.9487
Iter=96, loss=1.6492, rmse=5.5210, time=0.0565,	Val RMSE=2.9490
Iter=97, loss=1.6467, rmse=5.4877, time=0.0565,	Val RMSE=2.9494
Iter=98, loss=1.6445, rmse=5.5093, time=0.0565,	Val RMSE=2.9503
Iter=99, loss=1.6421, rmse=5.4968, time=0.0565,	Val RMSE=2.9513
Iter=100, loss=1.6399, rmse=5.4735, time=0.0565,	Val RMSE=2.9518
Iter=101, loss=1.6378, rmse=5.5362, time=0.0565,	Val RMSE=2.9516
Iter=102, loss=1.6356, rmse=5.4720, time=0.0565,	Val RMSE=2.9514
Iter=103, loss=1.6336, rmse=5.5153, time=0.0565,	Val RMSE=2.9518
Iter=104, loss=1.6315, rmse=5.5144, time=0.0565,	Val RMSE=2.9514
Iter=105, loss=1.6295, rmse=5.4557, time=0.0565,	Val RMSE=2.9517
Iter=106, loss=1.6275, rmse=5.5154, time=0.0565,	Val RMSE=2.9518
Iter=107, loss=1.6255, rmse=5.5172, time=0.0565,	Val RMSE=2.9524
Iter=108, loss=1.6236, rmse=5.4795, time=0.0565,	Val RMSE=2.9523
Iter=109, loss=1.6217, rmse=5.5099, time=0.0565,	Val RMSE=2.9523
Iter=110, loss=1.6198, rmse=5.4947, time=0.0565,	Val RMSE=2.9522
Iter=111, loss=1.6179, rmse=5.5390, time=0.0565,	Val RMSE=2.9526
Iter=112, loss=1.6162, rmse=5.5648, time=0.0565,	Val RMSE=2.9535
Iter=113, loss=1.6143, rmse=5.4625, time=0.0565,	Val RMSE=2.9546
Iter=114, loss=1.6126, rmse=5.4453, time=0.0565,	Val RMSE=2.9557
Iter=115, loss=1.6108, rmse=5.4732, time=0.0565,	Val RMSE=2.9562
Iter=116, loss=1.6090, rmse=5.4379, time=0.0565,	Val RMSE=2.9565
Iter=117, loss=1.6072, rmse=5.4306, time=0.0565,	Val RMSE=2.9557
Iter=118, loss=1.6055, rmse=5.4228, time=0.0565,	Val RMSE=2.9560
Iter=119, loss=1.6037, rmse=5.5061, time=0.0565,	Val RMSE=2.9570
Iter=120, loss=1.6021, rmse=5.5870, time=0.0565,	Val RMSE=2.9578
Iter=121, loss=1.6005, rmse=5.5190, time=0.0565,	Val RMSE=2.9585
Iter=122, loss=1.5989, rmse=5.4609, time=0.0565,	Val RMSE=2.9584
Iter=123, loss=1.5973, rmse=5.4371, time=0.0565,	Val RMSE=2.9576
Iter=124, loss=1.5957, rmse=5.4548, time=0.0565,	Val RMSE=2.9570
Iter=125, loss=1.5942, rmse=5.4504, time=0.0565,	Val RMSE=2.9565
Iter=126, loss=1.5927, rmse=5.4551, time=0.0565,	Val RMSE=2.9556
Iter=127, loss=1.5912, rmse=5.4983, time=0.0565,	Val RMSE=2.9554
Iter=128, loss=1.5898, rmse=5.4522, time=0.0565,	Val RMSE=2.9562
Iter=129, loss=1.5883, rmse=5.4612, time=0.0565,	Val RMSE=2.9574
Iter=130, loss=1.5869, rmse=5.5033, time=0.0565,	Val RMSE=2.9581
Iter=131, loss=1.5854, rmse=5.4489, time=0.0565,	Val RMSE=2.9580
Iter=132, loss=1.5840, rmse=5.4425, time=0.0565,	Val RMSE=2.9575
Iter=133, loss=1.5826, rmse=5.5091, time=0.0565,	Val RMSE=2.9572
Iter=134, loss=1.5812, rmse=5.4935, time=0.0565,	Val RMSE=2.9576
Iter=135, loss=1.5798, rmse=5.4211, time=0.0565,	Val RMSE=2.9578
Iter=136, loss=1.5784, rmse=5.4369, time=0.0565,	Val RMSE=2.9577
Iter=137, loss=1.5770, rmse=5.3794, time=0.0565,	Val RMSE=2.9579
Iter=138, loss=1.5757, rmse=5.4358, time=0.0565,	Val RMSE=2.9579
Iter=139, loss=1.5744, rmse=5.4255, time=0.0565,	Val RMSE=2.9579
Iter=140, loss=1.5731, rmse=5.4707, time=0.0565,	Val RMSE=2.9579
Iter=141, loss=1.5718, rmse=5.4181, time=0.0565,	Val RMSE=2.9580
Iter=142, loss=1.5706, rmse=5.3767, time=0.0565,	Val RMSE=2.9583
Iter=143, loss=1.5693, rmse=5.4873, time=0.0565,	Val RMSE=2.9588
Iter=144, loss=1.5681, rmse=5.4718, time=0.0565,	Val RMSE=2.9592
Iter=145, loss=1.5668, rmse=5.4135, time=0.0565,	Val RMSE=2.9595
Iter=146, loss=1.5656, rmse=5.3797, time=0.0565,	Val RMSE=2.9594
Iter=147, loss=1.5645, rmse=5.4165, time=0.0565,	Val RMSE=2.9593
Iter=148, loss=1.5633, rmse=5.4659, time=0.0565,	Val RMSE=2.9594
Iter=149, loss=1.5621, rmse=5.3761, time=0.0565,	Val RMSE=2.9597
Iter=150, loss=1.5609, rmse=5.4607, time=0.0565,	Val RMSE=2.9598
Iter=151, loss=1.5598, rmse=5.4941, time=0.0565,	Val RMSE=2.9600
Iter=152, loss=1.5587, rmse=5.4077, time=0.0565,	Val RMSE=2.9603
Iter=153, loss=1.5576, rmse=5.4610, time=0.0565,	Val RMSE=2.9604
Iter=154, loss=1.5565, rmse=5.4364, time=0.0565,	Val RMSE=2.9604
Iter=155, loss=1.5554, rmse=5.4227, time=0.0565,	Val RMSE=2.9603
Iter=156, loss=1.5543, rmse=5.4399, time=0.0565,	Val RMSE=2.9603
Iter=157, loss=1.5533, rmse=5.4132, time=0.0565,	Val RMSE=2.9603
Iter=158, loss=1.5522, rmse=5.4944, time=0.0565,	Val RMSE=2.9606
Iter=159, loss=1.5512, rmse=5.4348, time=0.0565,	Val RMSE=2.9611
Iter=160, loss=1.5501, rmse=5.3976, time=0.0565,	Val RMSE=2.9619
Iter=161, loss=1.5491, rmse=5.4898, time=0.0565,	Val RMSE=2.9628
Iter=162, loss=1.5481, rmse=5.4566, time=0.0565,	Val RMSE=2.9631
Iter=163, loss=1.5471, rmse=5.5112, time=0.0565,	Val RMSE=2.9631
Iter=164, loss=1.5462, rmse=5.4415, time=0.0565,	Val RMSE=2.9628
Iter=165, loss=1.5452, rmse=5.4176, time=0.0565,	Val RMSE=2.9629
Iter=166, loss=1.5442, rmse=5.4037, time=0.0565,	Val RMSE=2.9632
Iter=167, loss=1.5432, rmse=5.4718, time=0.0565,	Val RMSE=2.9635
Iter=168, loss=1.5422, rmse=5.4018, time=0.0565,	Val RMSE=2.9636
Iter=169, loss=1.5412, rmse=5.3735, time=0.0565,	Val RMSE=2.9638
Iter=170, loss=1.5403, rmse=5.4384, time=0.0565,	Val RMSE=2.9638
Iter=171, loss=1.5394, rmse=5.4319, time=0.0565,	Val RMSE=2.9639
Iter=172, loss=1.5384, rmse=5.3973, time=0.0565,	Val RMSE=2.9638
Iter=173, loss=1.5375, rmse=5.3730, time=0.0565,	Val RMSE=2.9638
Iter=174, loss=1.5367, rmse=5.4170, time=0.0565,	Val RMSE=2.9635
Iter=175, loss=1.5358, rmse=5.3760, time=0.0565,	Val RMSE=2.9634
Iter=176, loss=1.5349, rmse=5.4547, time=0.0565,	Val RMSE=2.9634
Iter=177, loss=1.5340, rmse=5.3842, time=0.0565,	Val RMSE=2.9637
Iter=178, loss=1.5331, rmse=5.5287, time=0.0565,	Val RMSE=2.9642
Iter=179, loss=1.5323, rmse=5.4413, time=0.0565,	Val RMSE=2.9649
Iter=180, loss=1.5314, rmse=5.4786, time=0.0565,	Val RMSE=2.9654
Iter=181, loss=1.5306, rmse=5.4481, time=0.0565,	Val RMSE=2.9655
Iter=182, loss=1.5297, rmse=5.3476, time=0.0565,	Val RMSE=2.9652
Iter=183, loss=1.5289, rmse=5.3917, time=0.0565,	Val RMSE=2.9650
Iter=184, loss=1.5281, rmse=5.4797, time=0.0565,	Val RMSE=2.9647
Iter=185, loss=1.5272, rmse=5.3323, time=0.0565,	Val RMSE=2.9648
Iter=186, loss=1.5264, rmse=5.3999, time=0.0565,	Val RMSE=2.9656
Iter=187, loss=1.5256, rmse=5.3254, time=0.0565,	Val RMSE=2.9665
Iter=188, loss=1.5247, rmse=5.3741, time=0.0565,	Val RMSE=2.9669
Iter=189, loss=1.5239, rmse=5.4145, time=0.0565,	Val RMSE=2.9675
Iter=190, loss=1.5231, rmse=5.3665, time=0.0565,	Val RMSE=2.9680
Iter=191, loss=1.5223, rmse=5.3904, time=0.0565,	Val RMSE=2.9681
Iter=192, loss=1.5216, rmse=5.4747, time=0.0565,	Val RMSE=2.9680
Iter=193, loss=1.5208, rmse=5.3606, time=0.0565,	Val RMSE=2.9678
Iter=194, loss=1.5201, rmse=5.4680, time=0.0565,	Val RMSE=2.9676
Iter=195, loss=1.5194, rmse=5.4377, time=0.0565,	Val RMSE=2.9674
Iter=196, loss=1.5187, rmse=5.4976, time=0.0565,	Val RMSE=2.9674
Iter=197, loss=1.5180, rmse=5.3850, time=0.0565,	Val RMSE=2.9674
Iter=198, loss=1.5172, rmse=5.3490, time=0.0565,	Val RMSE=2.9673
Iter=199, loss=1.5164, rmse=5.3713, time=0.0565,	Val RMSE=2.9672
Iter=200, loss=1.5157, rmse=5.3788, time=0.0565,	Val RMSE=2.9671
Iter=201, loss=1.5150, rmse=5.4366, time=0.0565,	Val RMSE=2.9670
Iter=202, loss=1.5143, rmse=5.3122, time=0.0565,	Val RMSE=2.9671
Iter=203, loss=1.5136, rmse=5.4218, time=0.0565,	Val RMSE=2.9672
Iter=204, loss=1.5129, rmse=5.3406, time=0.0565,	Val RMSE=2.9674
Iter=205, loss=1.5122, rmse=5.4310, time=0.0565,	Val RMSE=2.9677
Iter=206, loss=1.5116, rmse=5.3518, time=0.0565,	Val RMSE=2.9678
Iter=207, loss=1.5109, rmse=5.3709, time=0.0565,	Val RMSE=2.9679
Iter=208, loss=1.5101, rmse=5.3693, time=0.0565,	Val RMSE=2.9680
Iter=209, loss=1.5095, rmse=5.4259, time=0.0565,	Val RMSE=2.9680
Iter=210, loss=1.5088, rmse=5.4187, time=0.0565,	Val RMSE=2.9679
Iter=211, loss=1.5081, rmse=5.4102, time=0.0565,	Val RMSE=2.9677
Iter=212, loss=1.5075, rmse=5.4735, time=0.0565,	Val RMSE=2.9676
Iter=213, loss=1.5068, rmse=5.3551, time=0.0565,	Val RMSE=2.9674
Iter=214, loss=1.5061, rmse=5.3866, time=0.0565,	Val RMSE=2.9673
Iter=215, loss=1.5055, rmse=5.3713, time=0.0565,	Val RMSE=2.9671
Iter=216, loss=1.5048, rmse=5.3807, time=0.0565,	Val RMSE=2.9670
Iter=217, loss=1.5042, rmse=5.3636, time=0.0565,	Val RMSE=2.9669
Iter=218, loss=1.5036, rmse=5.4123, time=0.0565,	Val RMSE=2.9670
Iter=219, loss=1.5029, rmse=5.3891, time=0.0565,	Val RMSE=2.9672
Iter=220, loss=1.5023, rmse=5.3842, time=0.0565,	Val RMSE=2.9675
Iter=221, loss=1.5016, rmse=5.3915, time=0.0565,	Val RMSE=2.9678
Iter=222, loss=1.5011, rmse=5.4172, time=0.0565,	Val RMSE=2.9679
Iter=223, loss=1.5005, rmse=5.4287, time=0.0565,	Val RMSE=2.9679
Iter=224, loss=1.4999, rmse=5.3676, time=0.0565,	Val RMSE=2.9680
Iter=225, loss=1.4993, rmse=5.3880, time=0.0565,	Val RMSE=2.9681
Iter=226, loss=1.4987, rmse=5.3797, time=0.0565,	Val RMSE=2.9680
Iter=227, loss=1.4981, rmse=5.3672, time=0.0565,	Val RMSE=2.9680
Iter=228, loss=1.4976, rmse=5.4292, time=0.0565,	Val RMSE=2.9682
Iter=229, loss=1.4970, rmse=5.4000, time=0.0565,	Val RMSE=2.9683
Iter=230, loss=1.4965, rmse=5.3994, time=0.0565,	Val RMSE=2.9684
Iter=231, loss=1.4959, rmse=5.3977, time=0.0565,	Val RMSE=2.9685
Iter=232, loss=1.4954, rmse=5.4206, time=0.0565,	Val RMSE=2.9686
Iter=233, loss=1.4949, rmse=5.3733, time=0.0565,	Val RMSE=2.9688
Iter=234, loss=1.4943, rmse=5.3709, time=0.0565,	Val RMSE=2.9690
Iter=235, loss=1.4938, rmse=5.4155, time=0.0565,	Val RMSE=2.9689
Iter=236, loss=1.4932, rmse=5.3830, time=0.0565,	Val RMSE=2.9688
Iter=237, loss=1.4927, rmse=5.3979, time=0.0565,	Val RMSE=2.9689
Iter=238, loss=1.4921, rmse=5.3523, time=0.0565,	Val RMSE=2.9690
Iter=239, loss=1.4916, rmse=5.3480, time=0.0565,	Val RMSE=2.9690
Iter=240, loss=1.4910, rmse=5.3774, time=0.0565,	Val RMSE=2.9691
Iter=241, loss=1.4905, rmse=5.3680, time=0.0565,	Val RMSE=2.9692
Iter=242, loss=1.4900, rmse=5.3865, time=0.0565,	Val RMSE=2.9695
Iter=243, loss=1.4895, rmse=5.4497, time=0.0565,	Val RMSE=2.9696
Iter=244, loss=1.4890, rmse=5.4116, time=0.0565,	Val RMSE=2.9696
Iter=245, loss=1.4884, rmse=5.4161, time=0.0565,	Val RMSE=2.9696
Iter=246, loss=1.4879, rmse=5.4048, time=0.0565,	Val RMSE=2.9697
Iter=247, loss=1.4874, rmse=5.3149, time=0.0565,	Val RMSE=2.9697
Iter=248, loss=1.4869, rmse=5.3742, time=0.0565,	Val RMSE=2.9697
Iter=249, loss=1.4864, rmse=5.3517, time=0.0565,	Val RMSE=2.9698
Iter=250, loss=1.4859, rmse=5.3418, time=0.0565,	Val RMSE=2.9699
Iter=251, loss=1.4854, rmse=5.3881, time=0.0565,	Val RMSE=2.9700
Iter=252, loss=1.4849, rmse=5.3174, time=0.0565,	Val RMSE=2.9701
Iter=253, loss=1.4844, rmse=5.4367, time=0.0565,	Val RMSE=2.9704
Iter=254, loss=1.4839, rmse=5.3823, time=0.0565,	Val RMSE=2.9706
Iter=255, loss=1.4835, rmse=5.3793, time=0.0565,	Val RMSE=2.9706
Iter=256, loss=1.4830, rmse=5.3323, time=0.0565,	Val RMSE=2.9706
Iter=257, loss=1.4825, rmse=5.3924, time=0.0565,	Val RMSE=2.9704
Iter=258, loss=1.4820, rmse=5.3233, time=0.0565,	Val RMSE=2.9704
Iter=259, loss=1.4816, rmse=5.3841, time=0.0565,	Val RMSE=2.9705
Iter=260, loss=1.4812, rmse=5.4272, time=0.0565,	Val RMSE=2.9706
Iter=261, loss=1.4807, rmse=5.3732, time=0.0565,	Val RMSE=2.9707
Iter=262, loss=1.4803, rmse=5.3436, time=0.0565,	Val RMSE=2.9708
Iter=263, loss=1.4798, rmse=5.3755, time=0.0566,	Val RMSE=2.9709
Iter=264, loss=1.4793, rmse=5.3546, time=0.0566,	Val RMSE=2.9709
Iter=265, loss=1.4789, rmse=5.3513, time=0.0566,	Val RMSE=2.9710
Iter=266, loss=1.4785, rmse=5.4219, time=0.0566,	Val RMSE=2.9710
Iter=267, loss=1.4780, rmse=5.3543, time=0.0566,	Val RMSE=2.9711
Iter=268, loss=1.4776, rmse=5.3115, time=0.0566,	Val RMSE=2.9711
Iter=269, loss=1.4771, rmse=5.3133, time=0.0566,	Val RMSE=2.9712
Iter=270, loss=1.4767, rmse=5.3440, time=0.0566,	Val RMSE=2.9716
Iter=271, loss=1.4763, rmse=5.3430, time=0.0566,	Val RMSE=2.9719
Iter=272, loss=1.4758, rmse=5.3955, time=0.0566,	Val RMSE=2.9724
Iter=273, loss=1.4754, rmse=5.3582, time=0.0566,	Val RMSE=2.9727
Iter=274, loss=1.4750, rmse=5.3792, time=0.0566,	Val RMSE=2.9730
Iter=275, loss=1.4745, rmse=5.3325, time=0.0566,	Val RMSE=2.9733
Iter=276, loss=1.4741, rmse=5.3581, time=0.0566,	Val RMSE=2.9733
Iter=277, loss=1.4738, rmse=5.4191, time=0.0566,	Val RMSE=2.9733
Iter=278, loss=1.4734, rmse=5.3839, time=0.0566,	Val RMSE=2.9733
Iter=279, loss=1.4729, rmse=5.3584, time=0.0566,	Val RMSE=2.9734
Iter=280, loss=1.4725, rmse=5.2813, time=0.0566,	Val RMSE=2.9733
Iter=281, loss=1.4721, rmse=5.3529, time=0.0566,	Val RMSE=2.9731
Iter=282, loss=1.4717, rmse=5.4671, time=0.0566,	Val RMSE=2.9728
Iter=283, loss=1.4713, rmse=5.2679, time=0.0566,	Val RMSE=2.9727
Iter=284, loss=1.4709, rmse=5.3973, time=0.0566,	Val RMSE=2.9727
Iter=285, loss=1.4705, rmse=5.3233, time=0.0566,	Val RMSE=2.9728
Iter=286, loss=1.4701, rmse=5.3553, time=0.0566,	Val RMSE=2.9729
Iter=287, loss=1.4697, rmse=5.3788, time=0.0566,	Val RMSE=2.9731
Iter=288, loss=1.4693, rmse=5.3478, time=0.0566,	Val RMSE=2.9734
Iter=289, loss=1.4690, rmse=5.3125, time=0.0566,	Val RMSE=2.9737
Iter=290, loss=1.4686, rmse=5.3260, time=0.0566,	Val RMSE=2.9738
Iter=291, loss=1.4682, rmse=5.3412, time=0.0566,	Val RMSE=2.9738
Iter=292, loss=1.4678, rmse=5.3581, time=0.0566,	Val RMSE=2.9737
Iter=293, loss=1.4674, rmse=5.3447, time=0.0566,	Val RMSE=2.9734
Iter=294, loss=1.4670, rmse=5.3233, time=0.0566,	Val RMSE=2.9732
Iter=295, loss=1.4667, rmse=5.3620, time=0.0566,	Val RMSE=2.9729
Iter=296, loss=1.4663, rmse=5.3121, time=0.0566,	Val RMSE=2.9726
Iter=297, loss=1.4660, rmse=5.3392, time=0.0566,	Val RMSE=2.9722
Iter=298, loss=1.4656, rmse=5.3469, time=0.0566,	Val RMSE=2.9723
Iter=299, loss=1.4653, rmse=5.3559, time=0.0566,	Val RMSE=2.9724
Iter=300, loss=1.4649, rmse=5.3740, time=0.0566,	Val RMSE=2.9729
Iter=301, loss=1.4645, rmse=5.3656, time=0.0566,	Val RMSE=2.9734
Iter=302, loss=1.4642, rmse=5.2815, time=0.0566,	Val RMSE=2.9740
Iter=303, loss=1.4638, rmse=5.3442, time=0.0566,	Val RMSE=2.9746
Iter=304, loss=1.4635, rmse=5.4615, time=0.0566,	Val RMSE=2.9750
Iter=305, loss=1.4631, rmse=5.3479, time=0.0566,	Val RMSE=2.9751
Iter=306, loss=1.4628, rmse=5.3971, time=0.0566,	Val RMSE=2.9749
Iter=307, loss=1.4624, rmse=5.3485, time=0.0566,	Val RMSE=2.9747
Iter=308, loss=1.4621, rmse=5.3829, time=0.0566,	Val RMSE=2.9743
Iter=309, loss=1.4617, rmse=5.3448, time=0.0566,	Val RMSE=2.9741
Iter=310, loss=1.4614, rmse=5.3613, time=0.0566,	Val RMSE=2.9741
Iter=311, loss=1.4610, rmse=5.3862, time=0.0566,	Val RMSE=2.9742
Iter=312, loss=1.4607, rmse=5.3479, time=0.0566,	Val RMSE=2.9744
Iter=313, loss=1.4603, rmse=5.3596, time=0.0566,	Val RMSE=2.9746
Iter=314, loss=1.4600, rmse=5.3735, time=0.0566,	Val RMSE=2.9748
Iter=315, loss=1.4597, rmse=5.3313, time=0.0566,	Val RMSE=2.9750
Iter=316, loss=1.4593, rmse=5.3201, time=0.0566,	Val RMSE=2.9752
Iter=317, loss=1.4590, rmse=5.3820, time=0.0566,	Val RMSE=2.9754
Iter=318, loss=1.4586, rmse=5.3128, time=0.0566,	Val RMSE=2.9756
Iter=319, loss=1.4583, rmse=5.3283, time=0.0566,	Val RMSE=2.9757
Iter=320, loss=1.4580, rmse=5.4073, time=0.0566,	Val RMSE=2.9759
Iter=321, loss=1.4577, rmse=5.4061, time=0.0566,	Val RMSE=2.9760
Iter=322, loss=1.4573, rmse=5.2651, time=0.0566,	Val RMSE=2.9759
Iter=323, loss=1.4570, rmse=5.3251, time=0.0566,	Val RMSE=2.9760
Iter=324, loss=1.4567, rmse=5.2569, time=0.0566,	Val RMSE=2.9762
Iter=325, loss=1.4564, rmse=5.3094, time=0.0566,	Val RMSE=2.9766
Iter=326, loss=1.4560, rmse=5.3998, time=0.0566,	Val RMSE=2.9771
Iter=327, loss=1.4557, rmse=5.2658, time=0.0566,	Val RMSE=2.9773
Iter=328, loss=1.4554, rmse=5.2944, time=0.0566,	Val RMSE=2.9775
Iter=329, loss=1.4551, rmse=5.4027, time=0.0566,	Val RMSE=2.9776
Iter=330, loss=1.4547, rmse=5.2958, time=0.0566,	Val RMSE=2.9778
Iter=331, loss=1.4544, rmse=5.3471, time=0.0566,	Val RMSE=2.9778
Iter=332, loss=1.4541, rmse=5.3754, time=0.0566,	Val RMSE=2.9775
Iter=333, loss=1.4538, rmse=5.4312, time=0.0566,	Val RMSE=2.9772
Iter=334, loss=1.4535, rmse=5.3554, time=0.0566,	Val RMSE=2.9770
Iter=335, loss=1.4532, rmse=5.3472, time=0.0566,	Val RMSE=2.9769
Iter=336, loss=1.4529, rmse=5.4086, time=0.0566,	Val RMSE=2.9768
Iter=337, loss=1.4526, rmse=5.3726, time=0.0566,	Val RMSE=2.9768
Iter=338, loss=1.4523, rmse=5.4064, time=0.0566,	Val RMSE=2.9767
Best Iter Idx=34, Best Valid RMSE=2.9031, Best Test RMSE=2.9494, Total Time=19.0242
```
