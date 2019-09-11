import numpy as np


def verify_ts(C, h0, hstar, TS, pref):
    C1 = C

    possible_hs = [[] for _ in range(len(TS[hstar]) + 1)]
    possible_hs[0] = [h0]

    for i, ts in enumerate(TS[hstar]):
        if len(possible_hs[i]) == 1 and possible_hs[i][0] == hstar:
            return True

        indexes = (C1[:-1, ts[0]] == ts[1])
        indexes = np.append(indexes, np.array([True]))
        C1 = C1[indexes]
        left = C1[:-1, -1]
        if len(left) == 0:
            break
        for hi in possible_hs[i]:

            pref_hi = pref[hi]
            best = 110

            temp_hs = []
            for l in left:
                if pref_hi[l] < best:
                    temp_hs = [l]
                if pref_hi[l] == best:
                    if not (l in possible_hs[i + 1]):
                        temp_hs.append(l)

                best = min(pref_hi[l], best)

            for h in temp_hs:
                possible_hs[i + 1].append(h)

    if len(possible_hs[len(TS[hstar])]) == 1 and possible_hs[len(TS[hstar])][0] == hstar:
        return True
    else:
        return False
