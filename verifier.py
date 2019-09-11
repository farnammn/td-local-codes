import numpy as np

def verify_ts(C, h0, hstar, TS, pref):
    C1 = C

    if h0 == hstar:
        return True

    for ts in TS[hstar]:
        # print(C1)
        indexes = (C1[:-1,ts[0]] == ts[1])
        indexes = np.append(indexes, np.array([True]))
        C1 = C1[indexes]
        left = C1[:-1, -1]
        pref_h0 = pref[h0]
        # print(left)
        best = 110
        if len(left) == 0:
            break
        for l in left:
            if pref_h0[l] < best:
                h0 = l

            best = min(pref_h0[l], best)


        # print(h0)
        if h0 == hstar:
            return True

    return False
