import os
import pandas as pd

from PIL import Image, ImageStat
import imagehash
import cv2


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


def prepare_images(df, orig_dir_path, dest_dir_path):
    for filename in df["filename"]:
        try:
            filepath = os.path.join(orig_dir_path, filename)
            img = cleanup_picture(filepath, show_images=False)
            copy_filepath = os.path.join(dest_dir_path, filename)
            cv2.imwrite(copy_filepath,img)
            
        except Exception as e:
            print(f"Erreur lors de l'analyse de {filename}: {str(e)}")


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
    margin = 10
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