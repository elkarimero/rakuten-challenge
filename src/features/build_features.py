from image_features import extract_image_features
import pandas as pd

def main():

    # load raw train dataset
    X_train = pd.read_csv("../../data/raw/X_train.csv", sep=",",index_col=0)
    X_train["filename"] = "image_" + X_train["imageid"].astype(str) + "_product_" + X_train["productid"].astype(str) + ".jpg" # add filename for later processing

    # Extract text features
    # todo 

    # Extract image features
    img_train_rep = "/mnt/c/Users/karim/rakuten/images/data_clean/image_train"
    X_train = extract_image_features(X_train, img_train_rep)

    # Save processed features
    X_train.to_csv("../../data/processed/X_train_features.csv", index=False)

if __name__ == "__main__":
    main()