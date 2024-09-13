import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
from tqdm import tqdm
AMPLIFICATION_FACTOR = 0.4
Z_NUMBER = 2500
GAMMA = 2
root = "/data/luo/reproduce/privacy_and_aug"

with open("/home/luo/sampleinfo/samplelist.txt", "r") as f:
    samplelist = eval(f.read())

#with open("sampleinfo/target.txt", "r") as f:
    #samplelist_target = eval(f.read())



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training models.")

    parser.add_argument('--load_model', action='store_true', default=False)

    parser.add_argument('--load_best_model', action='store_true', default=False)

    parser.add_argument('--target_model',default=0, type=int, help='perturbation bound')

    parser.add_argument('--ref_model_num', default=0, type=int, help='perturbation bound')

    parser.add_argument('--mode', default="offline", choices=["online", "offline"])

    parser.add_argument('--aug_type', default="base", choices=["base", "pgdat"])

    return parser.parse_args()

allmodellist = list(range(0, 128))

args = get_arguments()

dirs = os.path.join(root, "phy", 'cifar100', args.aug_type)

TARGET_CONF = np.load("%s/conf_%s.npy" % (dirs, str(args.target_model)))

out_ref_indicator_index = {}
in_ref_indicator_index = {}
z_indicator_index = []
s_indicator_index_list = []


ref_out_model_list = []

indicator = np.zeros(60000, dtype=np.bool_)

def get_x_inference_point(modellist):
    cant_be_x_indicator_index = []
    for model_number in modellist:
        cant_be_x_indicator_index = set(cant_be_x_indicator_index).union(samplelist[model_number])

    all_indices = set(range(60000))  # 创建0到60000的所有索引

    # 将 cant_be_x_indicator_index 转换为集合
    cant_be_x_indicator_set = set(cant_be_x_indicator_index)
    possiable_x_indices = all_indices - cant_be_x_indicator_set
    return possiable_x_indices

def get_random_datapoints(total_points, num_samples):
    return random.sample(range(total_points), num_samples)

def get_each_ratio_x(x_index, ref_out_model_list, ref_in_model_list, target_model):
    dirs = os.path.join(root, "phy", 'cifar100', args.aug_type)
    prx_out = []
    prx_in = []
    for ref_out in ref_out_model_list:
        out_conf = np.load("%s/conf_%s.npy" % (dirs, str(ref_out)))
        prx_out.append(out_conf[x_index]) #that specifical datapoint

    prx_out = np.mean(prx_out)
    if args.mode == "online":
        for ref_in in ref_in_model_list:

            in_conf = np.load("%s/conf_%s.npy" % (dirs, str(ref_in)))
            prx_in.append(in_conf[x_index])

        prx_in = np.mean(prx_in)

    if args.mode == "online":
        prx = (prx_out + prx_in)/2
    else:
        prx = ((1+AMPLIFICATION_FACTOR)*prx_out + (1-AMPLIFICATION_FACTOR)) / 2


    return prx/TARGET_CONF[x_index]

def get_rmia_score(ratio_x,ref_out_model_list, ref_in_model_list,target_model):
    dirs = os.path.join(root, "phy", 'cifar100', args.aug_type)
    z_indicator_index_list = get_random_datapoints(60000, Z_NUMBER)
    ref_all_model_list = ref_out_model_list

    overall_count = 0

    for z_index in z_indicator_index_list:

        prz_conf = []
        for ref in ref_all_model_list:
            conf = np.load("%s/conf_%s.npy" % (dirs, str(ref)))
            prz_conf.append(conf[z_index])  # that specifical datapoint
        prz = np.mean(prz_conf)

        ratio_z = TARGET_CONF[z_index]/prz

        if ratio_x/ratio_z > GAMMA:
            overall_count = overall_count + 1

    return overall_count/Z_NUMBER

def get_offline_x(x_index, ref_out_model_list, ref_in_model_list, target_model):
    dirs = os.path.join(root, "phy", 'cifar100', args.aug_type)
    prx_out = []
    prx_in = []
    for ref_out in ref_out_model_list:
        out_conf = np.load("%s/conf_%s.npy" % (dirs, str(ref_out)))
        prx_out.append(out_conf[x_index]) #that specifical datapoint

    prx_out = np.mean(prx_out)
    if args.mode == "online":
        for ref_in in ref_in_model_list:

            in_conf = np.load("%s/conf_%s.npy" % (dirs, str(ref_in)))
            prx_in.append(in_conf[x_index])

        prx_in = np.mean(prx_in)

    if args.mode == "online":
        prx = (prx_out + prx_in)/2
    else:
        prx = ((1+AMPLIFICATION_FACTOR)*prx_out + (1-AMPLIFICATION_FACTOR)) / 2


    return prx/TARGET_CONF[x_index]



def check_one_point_in_out(point_index):
    print("kk")



def rmia():
    allmodellist.remove(args.target_model)
    rmia_score = []

    for i in range(args.ref_model_num):
        rand_num = random.randint(64, 128)
        if rand_num not in out_ref_indicator_index.values() and rand_num in allmodellist:
            ref_out_model_list.append(rand_num)


    x_indicator_index_list = get_x_inference_point(ref_out_model_list)

    s_indicator_index_list = samplelist[args.target_model]


    print(z_indicator_index)
    print(len(z_indicator_index))
    indicator = np.zeros(60000, dtype=np.bool_)
    indicator[samplelist[args.target_model]] = True
    final_indicator = []

    for x in tqdm(x_indicator_index_list, desc="Processing"):
        if indicator[x] == True:
            final_indicator.append(True)
        else:
            final_indicator.append(False)

        ratio_x = get_each_ratio_x(x, ref_out_model_list, None, args.target_model)
        rmia_score.append(get_rmia_score(ratio_x,ref_out_model_list, None, args.target_model))

    assert len(rmia_score) == len(x_indicator_index_list)

    np.save("mia/rmia_conf/%s_rmia_score.npy" % (args.target_model), rmia_score)
    np.save("mia/rmia_conf/target_model_%s_indicator_rmia_score.npy" % (args.target_model), final_indicator)


def lira():

    allmodellist.remove(args.target_model)
    rmia_score = []

    for i in range(args.ref_model_num):
        rand_num = random.randint(64, 128)
        if rand_num not in out_ref_indicator_index.values() and rand_num in allmodellist:
            ref_out_model_list.append(rand_num)


    x_indicator_index_list = get_x_inference_point(ref_out_model_list)

    s_indicator_index_list = samplelist[args.target_model]




if __name__ == "__main__":
    #rmia()
