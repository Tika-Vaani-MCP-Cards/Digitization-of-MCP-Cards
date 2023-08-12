import os
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='gt.txt', help='path to full log file')
args = argparser.parse_args()


pred_path = args.path
write_ptr = open(f'{pred_path.split(".")[0]}_processed.txt', 'w')
best_ptr = open(f'{pred_path.split(".")[0]}_best.txt', 'w')
lines = open(f'{pred_path}', 'r').readlines()
confi_index_dict = {}
name_confi_dict = {}
name_gt_dict = {}
name_to_line = {}
path_dict = {}
name_list = []

for line in lines:
    line1 = line.split(' ', 1)[-1]
    line_list  = line1.split(',')
    gt,pred,path,confi = line_list[0],line_list[1],line_list[2],float(line_list[3].split('_')[0])
    name = path.split('/')[-1].split('$')[0]+'.jpg'
    
    name_list.append(name)
    
    if name in confi_index_dict:
        confi_index_dict[name].append(confi)
        name_gt_dict[name].append([gt,pred])
        name_to_line[name].append(line)
        path_dict[name].append(path)

    else:
        confi_index_dict[name] = [confi]
        name_gt_dict[name] = [[gt,pred]]
        name_to_line[name] = [line]
        path_dict[name] = [path]

name_list = set(name_list)
for itr, name in enumerate(name_list):
    maxI = np.argmax(np.array(confi_index_dict[name]))
    gt,pred = name_gt_dict[name][maxI]
    line = name_to_line[name][maxI]
    write_ptr.write(line)
    best_ptr.write(path_dict[name][maxI]+'\n') # for azure and google
    
