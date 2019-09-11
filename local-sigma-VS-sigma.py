######################################
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx

# import sys, os
# sys.path.append('/code')
from worst_case_verifier import verify_ts



#import copy


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=120)
np.set_printoptions(threshold=np.nan)

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

##############Farnam CODES


def get_local_sigma(C):
    clen = len(C)
    local_sigmas = []
    l = [i for i in range(clen - 1)]

    for t in tuple(itertools.permutations(l)):
        for i in range(clen):



    tuple(itertools.permutations(l))
    [[0 for _ in range(clen)] for _ in range(clen)]




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


