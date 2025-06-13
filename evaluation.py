import numpy as np

def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        print(eav_count)
    return concord / total
