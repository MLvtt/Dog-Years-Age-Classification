import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def img_array_stack(full_mat, img_np):
    img_np_1d = img_np.ravel()
    return np.vstack((full_mat, img_np_1d))

def img2np(path, list_of_filenames, size=(500, 500)):
    pbar = tqdm(list_of_filenames)
    for i, fn in enumerate(pbar):
        if path[-1] == '/':
            fp = path + fn
        else:    
            fp = path + '/' + fn
        current_img = load_img(fp, target_size=size, color_mode='grayscale')
        img_np = img_to_array(current_img) * 1./255
        try:
            full_mat = img_array_stack(full_mat, img_np)
        except UnboundLocalError: 
            full_mat = img_np.ravel()
        # if i % 25 == 0:
            # print(round(i/len(list_of_filenames)*100,1)," percent complete         \r",)
    return full_mat

def img2np_save(path, outpath, img_size=(500, 500)):
    ds_name = path.split('/')[-1].split('_')[0].lower()
    filepaths = [x for x, y, z in os.walk(path)]
    categories = [y for x, y, z in os.walk(path)][0]
    img_lists = [z for x, y, z in os.walk(path)]
    # walk = [x for x in zip(filepaths[1:], categories, img_lists[1:])]
    for fp, cat, img_lst in zip(filepaths[1:], categories, img_lists[1:]):
        print(fp, cat, len(img_lst))
        full_mat = img2np(fp, img_lst, img_size)
        save_path = f"{outpath}{ds_name}_{cat.lower()}_{img_size[0]}_G.npy"
        np.save(save_path, full_mat)
        print(ds_name, cat.lower(), 'done')
    


if __name__ == "__main__":
    expert_ds_loc = '../../data/Expert_TrainEval'
    # petfinder_ds_loc = '../../data/PetFinder_All'
    output_path = '../data/'
    # img2np_save(expert_ds_loc, output_path, (224,224))
    # img2np_save(petfinder_ds_loc, output_path, (224,224))

    # expert_senior_imgs = [fn for fn in os.listdir(f'{expert_ds_loc}/Senior') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]
    # expert_adult_imgs = [fn for fn in os.listdir(f'{expert_ds_loc}/Adult') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]
    # expert_young_imgs = [fn for fn in os.listdir(f'{expert_ds_loc}/Young') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]

    # petfinder_senior_imgs = [fn for fn in os.listdir(f'{petfinder_ds_loc}/Senior') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]
    # petfinder_adult_imgs = [fn for fn in os.listdir(f'{petfinder_ds_loc}/Adult') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]
    # petfinder_young_imgs = [fn for fn in os.listdir(f'{petfinder_ds_loc}/Young') 
    #                         if fn.endswith('.jpg') or fn.endswith('.png')]

    # all_imgs = [expert_senior_imgs, expert_adult_imgs, expert_young_imgs, 
    #             petfinder_senior_imgs, petfinder_adult_imgs, petfinder_young_imgs]
    
    # labels = ['EX-Senior', 'EX-Adult', 'EX-Young', 
    #             'PF-Senior', 'PF-Adult', 'PF-Young']
    
    # ex_senior_500 = img2np(f'{expert_ds_loc}/Senior/', expert_senior_imgs)
    # np.save('../data/expert_senior_500.npy', ex_senior_500)
    # print('Expert-Senior Done')
    # ex_adult_224 = img2np(f'{expert_ds_loc}/Adult/', expert_adult_imgs, (224,224))
    # ex_young_224 = img2np(f'{expert_ds_loc}/Young/', expert_young_imgs, (224,224))


