import pandas as pd
import numpy as np

if __name__=='__main__':
    ner=pd.read_csv("./save/20230421-195856/linear_ner_nodev.csv")
    sa_list = []
    sa=pd.read_csv("./save/prompt_sa.csv")
    ans=pd.DataFrame({
        'id':ner['id'],
        'BIO_anno':ner['BIO_anno'],
        'class':sa['class']
    })
    # ans=pd.DataFrame(columns=['id','BIO_anno','class'])
    
    ans.to_csv("./save/ensemble5.csv",index=False)