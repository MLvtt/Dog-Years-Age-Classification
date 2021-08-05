import petpy
import pandas as pd
import urllib.request
from urllib.error import HTTPError
from datetime import date
import os, json, re, time
from tqdm.auto import tqdm, trange
from ast import literal_eval
from multiprocessing.pool import ThreadPool

### Connect to api
try:
    pf = petpy.Petfinder(key='Wh02Nn81tRMs4bVGMniq7f1KTNgFtPt8wCysKomBw4Bazpbqu8', 
                            secret='1033sFI9b2sgwvuQ9Q0sZKB7EjBvLQMpdLPbI86n')
    pf.breeds('dog')
except KeyError:
    pf = petpy.Petfinder(key='3iO98SzI9gtLCHpYVvl1KQJJR12pMz7WmK6f2K9QW2ErpXvDTu', 
                            secret='Vly6Ij9qJT9DYfgRI1AoemBOAHV2UVKV755sIgY0')


def get_puppy_pics_list_json(jsonloc='../data/full_puppy_list.json', limit=5, *args, **kwargs):
    breeds = pf.breeds('dog', return_df=True).name
    pbar = tqdm(breeds.values)
    full_puppy_list = []
    for breed in pbar:
        puppy_list = [{'id': x['id'], 'breed': breed, 'photo': x['primary_photo_cropped']['large']} 
                        for x in pf.animals(animal_type='dog',age='baby', breed=breed, *args, **kwargs)['animals'] 
                        if (x['primary_photo_cropped'] != []) and (x['primary_photo_cropped'] != None)]
        if len(puppy_list) > limit:
            tqdm.write(f"{breed} {len(puppy_list)}")
            full_puppy_list += puppy_list

    with open(jsonloc, 'w') as outfile:
        json.dump(full_puppy_list, outfile)

def download_puppy_pics(full_puppy_list):
    missed_rows = []
    unique_ids = set()
    pbar = tqdm(full_puppy_list)
    for row in pbar:
        id = row['id']
        breed = row['breed']
        img_url = row['photo']
        if id not in unique_ids:
            unique_ids.add(id)
            filepath = f'../data/Puppy/{id}_{breed}.jpg'
            try:
                urllib.request.urlretrieve(img_url,filepath)
                tqdm.write(f"{id} {breed} done")
            except (HTTPError, OSError):
                missed_rows.append(row)
                missed_count = len(missed_rows)
                tqdm.write(f"{id} {breed} missed!!! MISSED COUNT: {missed_count}")
                continue
        else:
            tqdm.write(f"{id} {breed} ALREADY IN SET")
    print(missed_rows)
    print(f"TOTAL IMAGES MISSED: {missed_count}")
    return missed_rows


if __name__ == '__main__':
    # get_puppy_pics_list_json('../data/full_puppy_list_50perpage.json', limit=2, results_per_page=50)
    
    with open ('../data/full_puppy_list_100perpage.json', 'r') as f:
        full_puppy_list = json.load(f) 
    
    missed_rows = download_puppy_pics(full_puppy_list)
    missed_rows = download_puppy_pics(missed_rows)
    missed_rows = download_puppy_pics(missed_rows)
