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

##############Farnam CODES

def get_1_out(C, i, l):
    out = []
    left = []

    if len(C) == 2:
        if C[0, i] == 1-l:
            left.append(list(C[0]))
            out.append(list(C[-1]))
            left.append(list(C[-1]))

            return np.array(out), np.array(left)


    for j in range(len(C) - 1):
        c = C[j]
        c1 = list(C[j])
        c1[i] = 1 - c[i]
        flag = 0
        for c2 in C:
            if np.array_equal(c1[:-1], c2[:-1]):
                if c[i] == l:
                    out.append(c)
                else:
                    left.append(c)
                flag = 1
                break
        if flag == 0:
            out.append(c)

    out.append(list(C[-1]))
    left.append(list(C[-1]))

    return np.array(out), np.array(left)

def find_smallest_C_dist_set(C):
    s = len(C[0]) - 1
    i = 0
    while i < s:
        _, left = get_1_out(C, i, 0)
        if len(left) == 1:
            C = np.concatenate((C[:, :i], C[:, (i+1):]), axis = 1)
            s = s - 1
        else:
            i = i + 1
    return(C)

def greedy_big_rtd(C):
    num = len(C[0]) - 1
    matrix_rtd_one = []
    prefs = []
    labels = []
    indexes_for_labels = []
    C1 = C
    i = 0
    # label_index = C1[0, -1]
    while True:

        size = len(C1) - 1
        if len(C1[0]) <= 2:
            labels.append(1-C1[0,0])
            indexes_for_labels.append(C1[-1,0])
            matrix_rtd_one.append(C1[0, -1])
            if size >= 2:
                matrix_rtd_one.append(C1[1, -1])
            if i == num - 1:
                return labels, indexes_for_labels, matrix_rtd_one
            return False

        lens = np.sum(C1[:-1, :-1], axis = 0)
        lens2 = size - lens
        l = np.argmin(lens)
        l2 = np.argmin(lens2)

        if lens[l] < lens2[l2]:
            # if i == 0:
            indexes = (C1[:-1, l] == 1)
            indexes = np.append(indexes, np.array([True]))
            label_index = C1[indexes][0, -1]
            matrix_rtd_one.append(label_index)

            labels.append(0)
            indexes_for_labels.append(C1[-1, l])
            indexes = (C1[:-1,l] == 0)
            indexes = np.append(indexes, np.array([True]))

            C1 = C1[indexes]
            C1 = np.concatenate((C1[:, :l], C1[:, (l + 1):]), axis=1)

        else:
            # if i == 0:
            indexes = (C1[:-1, l2] == 0)
            indexes = np.append(indexes, np.array([True]))
            label_index = C1[indexes][0, -1]
            matrix_rtd_one.append(label_index)

            labels.append(1)
            indexes_for_labels.append(C1[-1, l2])
            indexes = (C1[:-1, l2] == 1)
            indexes = np.append(indexes, np.array([True]))

            C1 = C1[indexes]
            C1 = np.concatenate((C1[:, :l2], C1[:, (l2 + 1):]), axis=1)
        i = i + 1




def find_local_pref(C, d):
    C = np.concatenate((C,  np.reshape(np.arange(len(C)), [len(C), 1]) ), axis =  1)
    C = np.concatenate((C,  np.reshape(np.arange(len(C[0])), [1, len(C[0])]) ), axis = 0)
    number = len(C) - 1


    TS = [[] for i in range(number)]
    prefrence = np.full((number, number), number + 1)
    graph = np.full((number, number), 0)
    groups = [[] for _ in range(d + 3)]

    group = C
    indexes = (group[:-1, -1] != 0)
    indexes = np.append(indexes, np.array([True]))
    group = group[indexes]

    groups[0].append((group, 0))


    for i in range(d+2):
        for index, group_pack in enumerate(groups[i]):

            group, label_index = group_pack


            if len(group) <= 1:
                continue
            if len(group[0]) <= 1:
                continue
            if len(group) == 2:
                # prefrence[past_index, group[0,-1]] = len(groups[i]) - index
                # graph[past_index, group[0,-1]] = 1
                # TS[group[0, -1]].append(ts)
                prefrence[label_index, group[0,-1]] = 0
                graph[label_index, group[0,-1]] = 1
                TS[group[0,-1]].append((group[-1,0], group[0,0]))

                continue

            group_revised = find_smallest_C_dist_set(group)
            if greedy_big_rtd(group_revised) != False:

                labels, indexes_l, matrix_rtd_one= greedy_big_rtd(group_revised)

                for index_c, c in enumerate(matrix_rtd_one):
                    prefrence[label_index, c] = len(matrix_rtd_one) - 1 - index_c
                    graph[label_index, c] = 1

                    indexes = (group_revised[:-1, -1] != c)
                    indexes = np.append(indexes, np.array([True]))
                    group_revised = group_revised[indexes]

                # if len(group_revised) <= 1:
                #     print("it's there")
                #     print(matrix_rtd_one)
                #     print(group)
                #     for index in range(len(labels)):
                #         l = labels[index]
                #         place = indexes_l[index]
                #         next_index = matrix_rtd_one[index]
                #
                #         ts = (place, 1 - l)
                #         TS[next_index].append(ts)
                #     continue

                group_temp = group_revised

                start = len(groups[i+1])
                for index in range(len(labels)):
                    l = labels[index]
                    place = indexes_l[index]
                    next_index = matrix_rtd_one[index]
                    p= np.where(group_temp[-1] ==place)[0][0]

                    group_temp, new_group = get_1_out(group_temp, p, l)

                    ts = (place, 1-l)
                    TS[next_index].append(ts)

                    if len(new_group) > 1 and len(new_group[0]) > 1:
                        groups[i+1].append((new_group, next_index))
                        for j in range(len(new_group) - 1):
                            TS[new_group[j, -1]].append(ts)
                    # else:
                    #     groups[i+1].append(([],next_index))

                    if index != len(labels) - 1:
                        group_temp = np.concatenate((group_temp[:, :p], group_temp[:, (p+1):]), axis=1)

                if len(matrix_rtd_one) > len(labels):
                    l = labels[-1]
                    place = indexes_l[-1]
                    next_index = matrix_rtd_one[-1]

                    ts = (place, l)
                    TS[next_index].append(ts)
                    if len(group_temp) > 1:
                        groups[i + 1].append((group_temp, next_index))
                        TS[group_temp[0,-1]].append(ts)
                        if group_temp[0,-1] == 90:
                            print("wtf")
                            print(l)
                            print(place)
                            print(next_index)
                            print(new_group)
                            print(group_temp)

                    continue



                # if len(group_temp) > 1:
                #     print("group_temp check", group_temp[0, -1])
                #     s = (group_revised[:-1,-1] ==group_temp[0, -1])
                #     s = np.append(s, np.array([True]))
                #     target = group_revised[s]
                #     target_last = list(target[-1])
                #     flag = 0
                #
                #     place2 = group_temp[-1,0]
                #
                #     for j in range(len(labels) - 2,-1,-1):
                #         place = indexes_l[j]
                #         p = np.where(target[-1] == place)[0][0]
                #
                #
                #         indexes = (C[:-1, -1] == matrix_rtd_one[j])
                #         indexes = np.append(indexes, np.array([True]))
                #         rtd_one_j = C[indexes]
                #
                #         if target[0, p] != labels[j] or rtd_one_j[0, place2] == group_temp[0,0]:
                #             group, past_index = groups[i+1][j + start]
                #             if rtd_one_j[0, place2] == group_temp[0,0]:
                #                 ts = (place2, group_temp[0,0])
                #             else:
                #                 ts = (place,1 - labels[j])
                #             TS[group_temp[0, -1]].append(ts)
                #             if len(group) == 0:
                #                 group = group_temp
                #             else:
                #                 last = list(group[-1])
                #                 for t in target_last:
                #                     if not (t in last):
                #                         indexes = (target[-1] != t)
                #                         target = target[:, indexes]
                #
                #                 group[-1] = list(target[0])
                #                 last = np.reshape(last,[1,len(last)])
                #                 group = np.concatenate((group, last), axis = 0)
                #
                #             flag = 1
                #             groups[i+1][j + start] = (group, past_index)
                #             break
                #
                #     if flag == 0:
                #         print("our greedy algorithm failed, I am sorry:)))")


            else:
                print("our greedy algorithm failed, I am sorry:)))")


    return prefrence, graph, TS, C

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

    type = "12-100"

    if type == "12-100":
        output_file = "12_100.txt"
        CLen = C16Len
        C = C16
    else:
        output_file = "5_10.txt"
        C = C14
        CLen = C14Len


    pref, graph,TS, revised_C = find_local_pref(C, 3)
    for i in range(CLen):
        verify_ts(revised_C, 0, i, TS, pref)

    with open(output_file, "w") as f:
        f.write("The concept class:")
        f.write("\n")
        f.write("H,X \t")
        for c in revised_C[-1,:-1]:
            f.write(str(c))
            f.write("\t")
        f.write("\n")
        for row in revised_C[:-1]:
            f.write("h{} \t".format(row[-1]))
            for c in row[:-1]:
                f.write(str(c))
                f.write("\t")
            f.write("\n")

        f.write("Preferance Function:")
        f.write("\n")
        f.write("\t")
        for i in range(CLen):
            f.write("h{}\t".format(i))
        f.write("\n")
        for i, preferance in enumerate(pref):
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


    show_graph_with_labels(graph)


