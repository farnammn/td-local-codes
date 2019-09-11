######################################
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx

# import sys, os
# sys.path.append('/code')
from worst_case_verifier import verify_ts



#import copy


# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4)
# np.set_printoptions(linewidth=120)
# np.set_printoptions(threshold=np.nan)

################################################################################
def rotate_forward(l, n):
    return l[-n:] + l[:-n]

def generate_from_template_C16():
    templateList = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]]
    # templateList = [[0, 1, 2],
    #                [2, 0, 4],
    #                [2, 0, 1]]
    C, numH, numX = generate_from_template(templateList)
    return C, numH, numX, [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return generate_from_template(templateList), len(templateList)
# enddef

def generate_sigma(C, numH, numX, metricFlag):
    S = np.zeros((numH, numH), dtype='int')
    if(metricFlag == 0):
        return S
    #endif
    for i in range(numH):
        for j in range(numH):
            # print("i=", i, " j=", j)
            # print("C[i, :]=", C[i, :])
            # print("C[j, :]=", C[j, :])
            S[i, j] = np.sum(np.abs(C[i, :] - C[j, :]))
        #endfor
    #endfor
    # print(type(S))
    # print(S.shape)
    return S
#enddef


def generate_from_template(templateList):
    Clist = []
    hashtable = {}
    ############################
    for h in templateList:
        strh = ''.join([str(x) for x in h])
        if (strh not in hashtable):
            hashtable[strh] = 1
            Clist.append(h)
        else:
            print("....", strh, " exists in hashtable\n")
    #endfor
    for htemp in templateList:
        for r in range(1, len(templateList[0]), 1):
            h = rotate_forward(htemp, r)
            strh = ''.join([str(x) for x in h])
            if(strh not in hashtable):
                hashtable[strh] = 1
                Clist.append(h)
            else:
                print("....", strh, " exists in hashtable\n")
        #endfor
    #endfor
    ############################
    C = np.array(Clist, dtype='int')
    return C, C.shape[0], C.shape[1]
#enddef




def generate_Cw14():
    Clist = [[1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 1, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 0, 1, 1],
         [0, 1, 1, 0, 1],
         [1, 0, 1, 1, 0],
         [1, 0, 1, 0, 1],
         [1, 1, 0, 1, 0]
         ]
    C = np.array(Clist, dtype='int')
    #print(type(C))
    #print(C.shape)
    return C, C.shape[0], C.shape[1], [0, 5]



def show_graph_with_labels(adjency_matrix):
    # rows, cols = np.where(adjacency_matrix == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    # gr = nx.Graph()
    # gr.add_edges_from(edges)
    # nx.draw(gr, node_size=500, with_labels=False)
    # plt.show()
    G = nx.from_numpy_matrix(adjency_matrix)
    nx.draw_networkx(G, with_labels=True)
    plt.show()




################################################################


def calculate_all_subclass(C):
    C = np.concatenate((C, np.reshape(np.arange(len(C)), [len(C), 1])), axis=1)
    C = np.concatenate((C, np.reshape(np.arange(len(C[0])), [1, len(C[0])])), axis=0)

    h = C.shape[0] - 1
    x = C.shape[1] - 1

    all_cs = [C]
    t = 0

    while t < len(all_cs):
        current_C = all_cs[t]
        for l in range(x):
            for y in range(2):
                indexes = (current_C[:-1, l] == y)
                indexes = np.append(indexes, np.array([True]))
                temp_C = current_C[indexes]


                if len(temp_C) == 1:
                    continue

                flag = 0
                for c in all_cs:
                    if np.array_equal(c, temp_C):
                        flag = 1
                        break

                if flag == 0:
                    all_cs.append(temp_C)
        t = t + 1
    return all_cs



# def greedy_td_best(C):
#     n = C.shape[1]
#     m = C.shape[0]
#     best_x1 = 0
#     best_x2 = 0
#     best_sign = 0
#     min = 101
#     for s in range(8):
#         for j in range(1, n):
#             for k in range(j , n):
#                 y1 = (s // 4) % 2
#                 y2 = (s // 2) % 2
#                 y3 = s % 2
#                 C[C[:, 0] == y1]


def neigbour(C):
    count = 0
    for j in range(len(C)):
        c = list(C[j])
        for i in range(10):
            temp_c = c
            temp_c[i] = 1 - c[i]
            for c2 in C:
                if np.array_equal(temp_c, c2):
                    count+=1
                    break

    print(count)



######################################
if __name__ == '__main__':
    ########### C16
    C16, C16_numH, C16_numX, C16_templates = generate_from_template_C16()
    C16_Sglobal = generate_sigma(C16, C16_numH, C16_numX, 0)
    C16_Smetric = generate_sigma(C16, C16_numH, C16_numX, 1)
    C16, numH, numX, Sg, Sm, templates = C16, C16_numH, C16_numX, C16_Sglobal, C16_Smetric, C16_templates
    C16Len = C16.shape[0]
    maxTD = 5

    C14, C14Len, _, _ = generate_Cw14()

    type = "5-10"

    if type == "12-100":
        output_file = "12_100.txt"
        CLen = C16Len
        C = C16
    else:
        output_file = "5_10.txt"
        C = C14
        CLen = C14Len


    # is_legal = [[1 for _ in range(C14Len)] for _ in range(C14Len)]
    # print(len(calculate_all_subclass(C)))
    # all_cs = calculate_all_subclass(C)
    neigbour(C16)


