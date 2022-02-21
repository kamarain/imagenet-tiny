import os
import matplotlib.pyplot as plt
import pickle

x_tr = []
y_tr = []
with open("../Files/DUMP_meta_clsloc.txt") as file_in:
    lines = []
    for line in file_in:
        line_el = line.split('{')
        class_num = line_el[1]
        class_num = class_num.split('}')[0]
        class_id = line_el[2]
        class_id = class_id.split('}')[0]

        im_dir_name = '/home/kamarain/Data/ImageNet-tiny/ImageNet-tiny-8x8-Gray-8bit-convType-3/ILSVRC/Data/CLS-LOC/train/'+class_id
        im_dir = os.listdir(im_dir_name)
        for im_num,im_file in enumerate(im_dir):
            print(f"Class num {class_num} - Image num {im_num}     ",end='\r')
            
            img = plt.imread(im_dir_name+'/'+im_file)
            x_tr.append(img)
            y_tr.append(class_num)
            
with open("training.dat", "wb") as f:
        pickle.dump([x_tr,y_tr], f)
