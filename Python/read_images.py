import os
import matplotlib.pyplot as plt
import pickle
import csv
import random

tr_save_x_file = "training_x.dat"
tr_save_y_file = "training_y.dat"
val_save_x_file = "validation_x.dat"
val_save_y_file = "validation_y.dat"


tr_dump_file = "../Files/DUMP_meta_clsloc.txt"
val_dump_file = "../Files/DUMP_validation.txt"


tiny_dir = "/home/kamarain/Data/ImageNet-tiny/ImageNet-tiny-8x8-Gray-8bit-convType-3/"

# 1.1 Check if training data needed
if os.path.isfile(tr_save_x_file):
    print(f"{tr_save_x_file} exists - overwrite?")
    uin = input("[Y/N]")
    if uin == "y" or uin == "Y":
        do_train = True
    else:
        do_train = False
else:
    do_train = True

#
# 1.1 Do training data
if do_train:
    x_tr = []
    y_tr = []

    with open(tr_dump_file) as file_in:
        lines = []
        for line in file_in:
            line_el = line.split('{')
            class_num = line_el[1]
            class_num = class_num.split('}')[0]
            class_id = line_el[2]
            class_id = class_id.split('}')[0]
            
            im_dir_name = os.path.join(tiny_dir, "ILSVRC/Data/CLS-LOC/train/",class_id)
            im_dir = os.listdir(im_dir_name)
            for im_num,im_file in enumerate(im_dir):
                print(f"Class num {class_num} - Image num {im_num}     ",end='\r')
            
                img = plt.imread(im_dir_name+'/'+im_file)
                x_tr.append(img)
                y_tr.append(class_num)
    with open(tr_save_x_file, "wb") as f:
        pickle.dump(x_tr, f)
    with open(tr_save_y_file, "wb") as f:
        pickle.dump(y_tr, f)
    print(f"Wrote training data to {tr_save_x_file} and {tr_save_y_file} - Done!")

else:
    print("Skipping training data")

#
# 2.1 Check if validation data needed
if os.path.isfile(val_save_x_file):
    print(f"{val_save_x_file} exists - overwrite?")
    uin = input("[Y/N]")
    if uin == "y" or uin == "Y":
        do_val = True
    else:
        do_val = False
else:
    do_val = True


    
#
# 2.2 Do validation data
if do_val:
    x_val = []
    y_val = []

    line_no = 0
    with open(val_dump_file) as file_in:
        line_no = line_no+1
        print(f"Generating validation data (this is pretty fast)")
        for line in file_in:
            line_el = line.split('{')
            file_prefix = line_el[1]
            file_prefix = file_prefix.split('}')[0]
            gt = line_el[2]
            gt = gt.split('}')[0]

            #im_file = '/home/kamarain/Data/ImageNet-tiny/ImageNet-tiny-8x8-Gray-8bit-convType-3/ILSVRC/Data/CLS-LOC/train/'+class_id
            im_file = os.path.join("foo",tiny_dir, "ILSVRC/Data/CLS-LOC/val/",
                                   file_prefix + "_tiny.bmp")
            img = plt.imread(im_file)
            x_val.append(img)
            y_val.append(gt)
                
    with open(val_save_x_file, "wb") as f:
        pickle.dump(x_val, f)
    with open(val_save_y_file, "wb") as f:
        pickle.dump(y_val, f)
    print(f"Wrote validation data to {val_save_x_file} and {val_save_y_file} - Done!")
else:
    print("Skipping validation data")

# 3. Write Kaggle files
with open('test.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['Id','Class'])

    # write the data
    for lineno in range(len(y_val)):
        writer.writerow([lineno+1, y_val[lineno]])
    print(f"Wrote Kaggle file test.csv!")
        
with open('sample_submission.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['Id','Class'])

    # write the data
    for lineno in range(len(y_val)):
        writer.writerow([lineno+1, random.randint(1,1000)])
    print(f"Wrote Kaggle file sample_submission.csv!")
