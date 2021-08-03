import splitfolders
in_path = '../data/Combo_Dataset/Adult_Senior_Only'
out_path = '../data/Train_Val_Test/CASO'
splitfolders.ratio(in_path, output=out_path, seed=1337, ratio=(.8, 0.15,0.05)) 
