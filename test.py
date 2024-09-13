import numpy as np
name = ["ivy","never full","diane","spd nano","carry all","boulogne","spd 20"]
rmb = [11809,12899,15459,11989,17729,15639,14159]
eur = [1490,1500,2000,1550,2300,2100,1800]

for i in range(len(eur)):
    fee = rmb[i] - eur[i]*7.9*0.82 - 700
    print(f"%s : %.2f" %(name[i],fee))