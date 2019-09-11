import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from worst_case_verifier import verify_ts

H = ["".join(seq[::-1]) for seq in itertools.product("01", repeat=7)]
print(H)
bin_dic = {}

for i, h in enumerate(H):
    bin_dic[h] = i

n = 7
m = len(H)

real_H = np.array([(list(seq)[::-1]) for seq in itertools.product([0,1], repeat=7)])
real_H = np.concatenate((real_H,  np.reshape(np.arange(m), [m, 1])), axis=1)
real_H = np.concatenate((real_H,  np.reshape(np.arange(n + 1), [1, n + 1])), axis=0)


TS = [[] for i in range(m)]
preference = np.full((m, m), m + 1) - np.eye(m) * (m+1)
graph = np.full((m, m), 0)


def rotate(l, i):
    return l[-i:] + l[:-i]


def rotate_ts(ts, i):
    ts_out = ((ts[0] + i) % n, ts[1])
    return ts_out


def add_pref_rotate(start, end, ts, pref):
    for i in range(n):
        graph[bin_dic[rotate(start, i)], bin_dic[rotate(end, i)]] = 1
        preference[bin_dic[rotate(start, i)], bin_dic[rotate(end, i)]] = pref

        for a in TS[bin_dic[rotate(start, i)]]:
            TS[bin_dic[rotate(end, i)]].append(a)
        TS[bin_dic[rotate(end, i)]].append(rotate_ts(ts, i))


def add_pref_rotate_end(start, end, ts):
    for i in range(n):
        graph[bin_dic[start], bin_dic[rotate(end, i)]] = 1
        preference[bin_dic[start], bin_dic[rotate(end, i)]] = i + 1

        for a in TS[bin_dic[start]]:
            TS[bin_dic[rotate(end, i)]].append(a)
        TS[bin_dic[rotate(end, i)]].append(rotate_ts(ts, i))


def add_pref(start, end, ts, pref):
    graph[bin_dic[start], bin_dic[end]] = 1
    preference[bin_dic[start], bin_dic[end]] = pref

    for a in TS[bin_dic[start]]:
        TS[bin_dic[end]].append(a)
    TS[bin_dic[end]].append(ts)


def show_graph_with_labels(adjency_matrix):
    G = nx.from_numpy_matrix(adjency_matrix)
    nx.draw_networkx(G, with_labels=H)
    plt.show()


add_pref_rotate_end('0000000', '1000000', (0, 1))
############################
add_pref('1000000', '1111111', (6,1), 6)
############################
add_pref_rotate('1000000', '1100000', (1,1), 1)
add_pref_rotate('1000000', '1110000', (2,1), 2)
add_pref_rotate('1000000', '1111000', (3,1), 3)
add_pref_rotate('1000000', '1111100', (4,1), 4)
add_pref_rotate('1000000', '1111110', (5,1), 5)
############################
add_pref_rotate('1100000', '1101000', (3,1), 1)
add_pref_rotate('1100000', '1101100', (4,1), 2)
add_pref_rotate('1100000', '1110100', (2,1), 3)
add_pref_rotate('1100000', '1100010', (5,1), 4)
add_pref_rotate('1100000', '1100101', (6,1), 5)
############################
add_pref_rotate('1110000', '1010000', (1,0), 1)
add_pref_rotate('1110000', '1010100', (4,1), 2)
add_pref_rotate('1110000', '1010110', (5,1), 3)
add_pref_rotate('1110000', '1111010', (3,1), 4)
add_pref_rotate('1110000', '1011101', (6,1), 5)
############################
add_pref_rotate('1111000', '1001000', (1,0), 1)
add_pref_rotate('1111000', '1001100', (4,1), 2)

show_graph_with_labels(graph)
output_file = "7_128.txt"

for i in range(m):
    print(verify_ts(real_H, 0, i, TS, preference))


with open(output_file, "w") as f:
        f.write("The concept class:")
        f.write("\n")
        f.write("H,X \t")
        for h in real_H[-1,:-1]:
            f.write(str(h))
            f.write("\t")
        f.write("\n")
        for row in real_H[:-1]:
            f.write("h{} \t".format(row[-1]))
            for c in row[:-1]:
                f.write(str(c))
                f.write("\t")
            f.write("\n")

        f.write("Preferance Function:")
        f.write("\n")
        f.write("\t")
        for i in range(m):
            f.write("h{}\t".format(i))
        f.write("\n")
        for i, preferance in enumerate(preference):
            f.write("h{}\t".format(i))
            for p in preferance:
                f.write(str(p)+"\t")
            f.write("\n")

        f.write("Teaching Sequence:\n")
        for i, ts in enumerate(TS):
            f.write("h{}\t".format(i))
            for t in ts:
                f.write("(x{}, {})\t".format(t[0], t[1]))
            f.write("\n")
