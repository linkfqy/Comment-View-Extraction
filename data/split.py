import pandas as pd

if __name__ == '__main__':
    original = pd.read_csv("train_data_public.csv")
    # original=original.iloc[:100,:]
    train = original.sample(frac=0.9, random_state=1)
    dev = pd.concat([original, train]).drop_duplicates(["id"], keep=False)
    train.to_csv("train_splited.csv", index=False)
    dev.to_csv("dev_splited.csv", index=False)
    """ # clusering （X）
    train_df = pd.read_csv("train_splited.csv")
    dev_df = pd.read_csv("dev_splited.csv")

    temp_train_df = train_df.sort_values(by='class',inplace=False,ascending=True)
    temp_dev_df = dev_df.sort_values(by='class',inplace=False,ascending=True)
    
    temp_train_df.to_csv("train_splited_clusering.csv", index=False)
    temp_dev_df.to_csv("dev_splited_clusering.csv", index=False) """
    