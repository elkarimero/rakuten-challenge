from image_features import extract_image_features, prepare_images, remove_na, generate_phash
import pandas as pd
import time

def main():

    # load raw train dataset
    X_train = pd.read_csv("../../data/raw/X_train.csv", sep=",",index_col=0)
    Y_train = pd.read_csv("../../data/raw/Y_train.csv", sep=",",index_col=0)
    X_train["filename"] = "image_" + X_train["imageid"].astype(str) + "_product_" + X_train["productid"].astype(str) + ".jpg" # add filename for later processing
    train_pictures = pd.concat([X_train[["filename"]], Y_train], axis=1)

    # Extract text features
    # todo 

    ##############
    # Images preprocessing
    ##############

    img_train_rep = "/mnt/c/Users/karim/rakuten/images/data_raw/image_train"
    #img_train_cleaned_rep = "../../data/processed/image_train"
    img_train_cleaned_rep = "/mnt/c/Users/karim/rakuten/images/data_clean/image_train"

    # Zoom images
    print("start: zoom images")
    start = time.time()
    #prepare_images(train_pictures, img_train_rep, img_train_cleaned_rep)
    end = time.time()
    print("temps d'exécution:",end-start)

    # Extract image features
    print("start: Extract image features")
    start = time.time()
    train_pictures = extract_image_features(train_pictures, img_train_cleaned_rep)
    train_pictures.to_csv("../../data/interim/train_pictures.csv")
    end = time.time()
    print("temps d'exécution:",end-start) 

    # Clean up
    start = time.time()
    #train_pictures = remove_na(train_pictures, img_train_rep)
    #train_pictures = train_pictures.drop_duplicates(subset=["hash"])
    end = time.time()
    print("temps d'exécution:",end-start)

    

    # Save processed features
    #train_pictures.to_csv("../../data/processed/train_pictures.csv")
    train_pictures[["filename", "hash"]].to_csv("../../data/processed/X_train_pictures.csv")
    train_pictures[["prdtypecode"]].to_csv("../../data/processed/Y_train_pictures.csv")

if __name__ == "__main__":
    main()