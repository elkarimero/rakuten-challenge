import os
from PIL import Image, ImageStat
import imagehash

import pandas as pd

def extract_image_features(df, directory_path):
    
    dataset_stats = {
        'imageid': [],
        'productid': [],
        'height': [], 
        'width': [], 
        'modes': [], 
        'formats': [], 
        'ratios': [], 
        'file_sizes': [],
        'mean_luminosity':[],
        'mean_stddev_luminosity': [],
        "hash": []
        }
    
    df["filename"] = "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"

    for filename in df["filename"]:
        try:

            filepath = os.path.join(directory_path, filename)
            image = Image.open(filepath)

            # extract technical infos about the image
            dataset_stats['height'].append(image.size[0] )
            dataset_stats['width'].append(image.size[1])
            dataset_stats['modes'].append(image.mode)
            dataset_stats['formats'].append(image.format)
            dataset_stats['ratios'].append(image.size[0] / image.size[1])
            dataset_stats['file_sizes'].append(os.path.getsize(filepath))
            
            # analyse luminosity
            image_stat = ImageStat.Stat(image)
            mean_luminosity = sum(image_stat.mean)/3
            mean_stddev_luminosity = sum(image_stat.stddev)/3
            dataset_stats['mean_luminosity'].append(mean_luminosity)
            dataset_stats['mean_stddev_luminosity'].append(mean_stddev_luminosity)
            
            # Generate a perceptual hash to make duplicates analysis easier
            phash = str(imagehash.phash(image))
            dataset_stats['hash'].append(phash)
            
        except Exception as e:
            print(f"Erreur lors de l'analyse de {filename}: {str(e)}")

    df_analyzed_img = pd.DataFrame(dataset_stats)       
    return pd.concat(df, df_analyzed_img)


