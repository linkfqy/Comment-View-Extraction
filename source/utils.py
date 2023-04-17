import numpy as np
label_list = ["B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", "O",
              "B-COMMENTS_N", "I-COMMENTS_N", "B-COMMENTS_ADJ", "I-COMMENTS_ADJ",]
label_number = len(label_list)
class_number = 3
label_dict = {}
for i, label in enumerate(label_list):
    label_dict[label] = i


def bio2ids(bio):
    return label_dict[bio]


def ids2bio(ids):
    return label_list[ids]


class AvgCalc:
    def __init__(self) -> None:
        self.sum = 0
        self.num = 0

    def put(self, x):
        self.sum += x
        self.num += 1

    def get_avg(self):
        return self.sum/self.num


def f1(S: set, G: set):
    # assert len(S) != 0 and len(G) != 0, f'EMPTY SET!!!S={S},G={S}'
    inter = S.intersection(G)
    if len(inter)==0:
        return 0
    P = len(inter)/len(S)
    R = len(inter)/len(G)
    return (2*P*R)/(P+R)


def kappa(confusion_matrix: np.ndarray):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)


def ner2set(id, ner: np.ndarray):
    '''
    默认ner的头尾都是'O'
    '''
    s = set()
    last_class = 'O'
    first_pos = 0
    for i in range(len(ner)):
        bio = ids2bio(ner[i])
        if bio[0] == 'B':
            if last_class != 'O':
                s.add((id, first_pos, i-1, last_class))
            first_pos = i
            last_class = bio[2:]
        if bio[0] == 'O':
            if last_class != 'O':
                s.add((id, first_pos, i-1, last_class))
            last_class = 'O'
        if bio[0] == 'I':
            last_class = bio[2:]
    return s


if __name__ == '__main__':
    balance_matrix = np.array(
        [
            [2,  1,  1],
            [1,  2,  1],
            [1,  1,  2]
        ]
    )
    print(f"{kappa(balance_matrix):.6f}")
    
    s1=ner2set(1,[4,4,0,1,1,4,4,0,1,5,7,4])
    s2=ner2set(1,[4,4,0,1,4,4,4,0,1,5,7,4])
    print(s1)
    print(s2)
    print(f1(s1,s2))
    s1.update(s2)
    print(s1)
    print(s1.intersection(s2))
