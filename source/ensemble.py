import pandas as pd

if __name__=='__main__':
    ner=pd.read_csv("./save/linear_ner_nodev.csv")
    sa=pd.read_csv("./save/gruattn_sa.csv")
    ans=pd.DataFrame({
        'id':ner['id'],
        'BIO_anno':ner['BIO_anno'],
        'class':sa['class']
    })
    ans.to_csv("./save/ensemble.csv",index=False)