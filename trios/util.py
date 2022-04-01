import numpy as np

#ctypedef fused raw_data:
#    cython.uchar
#    cython.uint
#    cython.double


def copy_data(X, i, win_size, pat):

    for j in range(win_size):
        X[i, j] = pat[j]


def dataset_to_array(dataset, dtype, include_output=True, minimize=False):
    npatterns = len(dataset)
    win_size = len(next(iter(dataset.keys())))
    i = 0
    ncols = win_size
    nrows = 0
    y = 0
    val_min=0

    for pat in dataset:
        occur = dataset[pat]
        nrows += len(occur) 
        
    X = np.zeros((nrows, ncols), dtype)
    C = np.zeros(nrows, np.uint32)
    if include_output:
        y = np.zeros(nrows, dtype=np.uint8)
    
    for pat in dataset:
        occur = dataset[pat]
        if minimize:
            val_min = sum([x * occur[x] for x in occur])/sum(occur.values())
        
        values = sorted(dataset[pat].keys())
        for val in values:            
            noccur = occur[val]            

            copy_data(X, i, win_size, pat)

                
            C[i] = noccur
            if include_output:
                if minimize:
                    y[i] = val_min
                else:
                    y[i] = val
            i += 1
        

    assert i == nrows
    if include_output:
        return X, y, C
    return X, C


def expand_dataset(X, y, C):
    nrows = sum(C)
    X2 = np.zeros((nrows, X.shape[1]), X.dtype)    
    y2 = np.zeros(nrows, y.dtype)
    
    k = 0
    for i in range(X.shape[0]):
        freq = C[i]
        for j in range(freq):
            X2[k+j] = X[i]
            y2[k+j] = y[i]
        k += freq
        
    assert k == nrows
    return X2, y2
    


def minimize_error(dataset):
    npatterns = len(dataset)
    win_size = len(next(iter(dataset.keys())))

    decisions = np.zeros((npatterns, win_size+1), dtype=np.uint32)
    freqs = np.zeros((npatterns, 2), dtype=np.uint32)

    keys_r = [tuple(reversed(k)) for k in dataset.keys()]
    keys_r.sort()
    #patterns = sorted(dataset.keys(), key=reversed)
    
    i = 0
    for pat_r in keys_r:
        fq_tot = fq0 = fq1 = 0
        pat = tuple(reversed(pat_r))
        if 0 in dataset[pat]:
            fq0 = dataset[pat][0]
        if 255 in dataset[pat]:
            fq1 = dataset[pat][255]
            
        assert fq0 != 0 or fq1 != 0
        
        fq_tot = fq0 + fq1
        freqs[i, 0] = fq_tot
        freqs[i, 1] = fq1
        
        for j in range(win_size):
            decisions[i,j] = pat[j] & 0xffffffff
            
        if fq0 >= fq1:
            decisions[i, win_size] = 0 & 0xffffffff
        else:
            decisions[i, win_size] = 1 & 0xffffffff
        i+=1
    return decisions, freqs
