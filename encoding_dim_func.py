from statsmodels.multivariate.pca import PCA


def find_N(ic, n):
    finding = 0
    for i in range(ic.shape[0]):
        if i == 0:
            pass
        else:
            x1 = ic.iloc[i - 1, n]
            x2 = ic.iloc[i, n]
            if x1 < x2:
                finding += i
                break
    return finding


def Find_factor_number(data, ic_n):
    # ic_n = {0, 1, 2}
    pc = PCA(data)
    ic = pc.ic
    number = find_N(ic, ic_n)
    return number
