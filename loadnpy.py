import numpy as np
results = [0,8,13,14,16,47,53,77,91,92,93,97,102,114,117,120,123,126,127]
trades = np.load("trades_notconverage.npy")
tradesfat=np.load("trades_fat_notconverage.npy")

np.save(f"trades_notconverage.npy", results)
resultsnot = [0, 1 ,2 ,3 ,4 ,5 ,6 ,8,10,11,12,13,14,15,18,20,23,25,28,

 29, 31, 33, 35 ,36 ,37 ,38 ,45, 46 ,47 ,49 ,52, 54 ,57 ,61, 63 ,65, 69,

 71 ,72 ,73, 74, 76 ,79, 84,86, 88 ,91 ,92 ,95, 96 ,98, 99 ,103 ,104 ,105,

 106, 109 ,111, 116 ,120, 122, 124 ,125 ,126]
np.save(f"trades_fat_notconverage.npy", resultsnot)