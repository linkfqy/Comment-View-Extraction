import pandas as pd

if __name__ == '__main__':
    original = pd.read_csv("train_data_public.csv")
    # original=original.iloc[:100,:]
    train = original.sample(frac=0.9, random_state=1)
    dev = pd.concat([original, train]).drop_duplicates(["id"], keep=False)
    train.to_csv("train_splited.csv", index=False)
    dev.to_csv("dev_splited.csv", index=False)