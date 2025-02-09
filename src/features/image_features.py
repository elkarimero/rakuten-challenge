import os
import pandas as pd
import numpy as np

from PIL import Image, ImageStat
import imagehash
import cv2

def generate_phash(filename, directory_path):
    try:
        filepath = os.path.join(directory_path, filename)
        image = Image.open(filepath)
        return str(imagehash.phash(image))
    except Exception as e:
        print(f"Erreur lors de l'analyse de {filename}: {str(e)}")
        return np.nan

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
        "hash": [],
        "mean_std": []
        }
    data = []

    for filename in df["filename"]:
        try:

            filepath = os.path.join(directory_path, filename)
            image = Image.open(filepath)

            # extract technical infos about the image
            # dataset_stats['height'].append(image.size[0] )
            # dataset_stats['width'].append(image.size[1])
            # dataset_stats['modes'].append(image.mode)
            # dataset_stats['formats'].append(image.format)
            # dataset_stats['ratios'].append(image.size[0] / image.size[1])
            # dataset_stats['file_sizes'].append(os.path.getsize(filepath))
            
            # analyse luminosity
            #image_stat = ImageStat.Stat(image)
            #mean_luminosity = sum(image_stat.mean)/3
            #mean_stddev_luminosity = sum(image_stat.stddev)/3
            #dataset_stats['mean_luminosity'].append(mean_luminosity)
            #dataset_stats['mean_stddev_luminosity'].append(mean_stddev_luminosity)
            
            # Generate a perceptual hash to make duplicates analysis easier
            phash = str(imagehash.phash(image))
            #dataset_stats['hash'].append(phash)

            # standard deviation analysis
            #image_array = cv2.imread(filepath)
            #image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            #R, G, B = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
            #std_R = np.std(R)
            #std_G = np.std(G)
            #std_B = np.std(B)
            #dataset_stats['mean_std'].append(np.mean([std_R, std_G, std_B]))
            
            row = {
                "hash": phash
            }

            data.append(row)

        except Exception as e:
            print(f"Erreur lors de l'analyse de {filename}: {str(e)}")
            row = {
                "hash": np.nan
            }
            data.append(row)

    df_analyzed_img = pd.DataFrame(data)       
    return df_analyzed_img #pd.concat(df, df_analyzed_img)


def prepare_images(df, orig_dir_path, dest_dir_path):
    for filename in df["filename"]:
        try:
            filepath = os.path.join(orig_dir_path, filename)
            img = cleanup_picture(filepath, show_images=False)
            copy_filepath = os.path.join(dest_dir_path, filename)
            cv2.imwrite(copy_filepath,img)
            
        except Exception as e:
            print(f"Erreur dans prepare_images pour {filename}: {str(e)}")


def cleanup_picture(filepath, show_images = True):
    img_src = cv2.imread(filepath)
    image = img_src.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Apply threshold to binarize the image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the biggest bounding box detected
    max_size = 0
    x_max, y_max, w_max, h_max = 0, 0, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_size = w*h
        if contour_size > max_size: 
            max_size = contour_size
            x_max, y_max, w_max, h_max = x, y, w, h

    # Add margin to bounding box 
    margin = 1 
    image_width, image_height = 500, 500
    x = max(0, x_max - margin)
    w = min(w_max + 2 * margin, image_width - x)
    y = max(0, y_max - margin)
    h = min(h_max + 2 * margin, image_height - y)

    # draw the bounding box on original picture
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # crop picture to eliminate white background
    cropped_image = img_src[y:y+h, x:x+w]

    # find ratio to resize properly
    scale = min(image_width / w, image_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped_image, (new_w, new_h))

    return resized

def remove_na(data, src_dir_path):

    # pictures identified as na
    na_pictures = [
        'image_1243417369_product_3773721822.jpg',
        'image_1026011984_product_558486759.jpg', 
        'image_1190250938_product_2832752183.jpg', 
        'image_1268286826_product_3948210877.jpg', 
        'image_1190252023_product_2647272665.jpg', 
        'image_1261394799_product_3898719785.jpg', 
        'image_977145542_product_277513729.jpg', 
        'image_1306155830_product_4164869671.jpg', 
        'image_1100102350_product_1712289008.jpg',
        'image_1142089742_product_884747735.jpg'
        ]
    
        # image_1248838417_product_3817897828.jpg
        # image_1248834760_product_3817892731.jpg
        
    
    # Generate a perceptual hash to identifier similare pictures in the dataset
    na_hash = []
    for filename in na_pictures: 
        filepath = os.path.join(src_dir_path, filename)
        phash = str(imagehash.phash(Image.open(filepath)))
        na_hash.append(phash)

    
    # Generate a perceptual hash for all pictures to identify na in the dataset
    pictures_hash = []
    for filename in data["filename"]: 
        pictures_hash.append(generate_phash(filename, src_dir_path))

    data["hash"] = pictures_hash

    # Remove nan
    return data[~data["hash"].isin(na_hash)]



# na ?
# ffff80aad00a80ab 
# e66699e1e65a1895 
# b962df80a20dd4eb
# bb30f0ca0fcfc270
# b86bc3903c8f9666
# d5f5d4926a432cd4
# ead4956a956a952a
# eae8953b953f8540

# 87347a3d6bc96485
# 87347a2f3f8d3085

# e141aeaed134d933