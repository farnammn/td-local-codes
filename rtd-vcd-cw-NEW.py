######################################
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx
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


def find_smallest_c_dist_set(C):
    s = len(C[0]) - 1
    i = 0
    while i < s:
        _, left = get_1_out(C, i, 0)
        if len(left) == 1:
            C = np.concatenate((C[:, :i], C[:, (i+1):]), axis = 1)
            s -= 1
        else:
            i += 1
    return(C)


def find_local_pref(C, d, c0_id):
    C = np.concatenate((C,  np.reshape(np.arange(len(C)), [len(C), 1]) ), axis =  1)
    C = np.concatenate((C,  np.reshape(np.arange(len(C[0])), [1, len(C[0])]) ), axis = 0)
    number = len(C) - 1

    TS = [[] for _ in range(number)]
    preference = np.full((number, number), number + 1)
    graph = np.full((number, number), 0)
    groups = [[] for _ in range(d + 2)]

    group = C
    indexes = (group[:-1, -1] != c0_id)
    indexes = np.append(indexes, np.array([True]))
    group = group[indexes]
    # print(C[c0])

    groups[0].append((group, c0_id))

    for i in range(d+1):
        for index, group_pack in enumerate(groups[i]):
            group, current_h_id = group_pack

            if len(group) <= 1:
                continue
            if len(group[0]) <= 1:
                continue
            if len(group) == 2:
                preference[current_h_id, group[0,-1]] = 0
                graph[current_h_id, group[0,-1]] = 1
                TS[group[0,-1]].append((group[-1,0], group[0,0]))
                continue

            group_revised = find_smallest_c_dist_set(group)

            last_column = group_revised[-1]
            current_h = C[current_h_id]
            current_h = current_h[last_column]
            length = len(current_h) - 1
            matrix_rtd_one = [0 for _ in range(length)]
            positions = [0 for _ in range(length)]

            C1 = group_revised
            current_h_C1 = current_h

            for j in range(length):
                flag = False
                print(C1)
                print(current_h_C1)
                for k in range(len(current_h_C1) - 1):
                    check = current_h_C1
                    check[k] = 1 - current_h_C1[k]
                    places = np.where((C1 == check).all(axis = 1))[0]
                    print(places)
                    if len(places) > 0:
                        p = places[0]
                        label_index = C1[p, -1]
                        min_index = k
                        position = C1[-1, min_index]

                        positions[length - 1 - j] = position
                        matrix_rtd_one[length - 1 - j] = label_index
                        flag = True
                        break

                if not flag:
                    return "algorithm failed"

                # X = C1[:-1, :-1] * 2 - 1
                # Y = 2 * current_h_C1[:-1] - 1
                # X = (np.multiply(X, Y) + 1) / 2
                # lens = np.sum(X, axis= 0)
                #
                # min_index = np.argmax(lens)
                # positions.append(C1[-1, min_index])


                # indexes = (C1[:-1, min_index] == 1 - current_h_C1[min_index])
                # indexes = np.append(indexes, np.array([True]))
                # if sum(indexes) == 1:
                #     return "algorithm failed"
                # label_index = C1[indexes][0, -1]



                # indexes = (C1[:-1, min_index] == current_h_C1[min_index])
                # indexes = np.append(indexes, np.array([True]))
                # C1 = C1[indexes]
                C1 = np.concatenate((C1[:, :min_index], C1[:, (min_index + 1):]), axis=1)
                current_h_C1 = np.concatenate((current_h_C1[:min_index] ,current_h_C1[(min_index + 1):]))

                print(C1)
                print(current_h_C1)


            for j, label_index in enumerate(matrix_rtd_one):
                preference[current_h_id, label_index] = length - j
                graph[current_h_id, label_index] = 1

                indexes = (group_revised[:-1, -1] != label_index)
                indexes = np.append(indexes, np.array([True]))
                group_revised = group_revised[indexes]

            group_temp = group_revised
            for index in range(length):
                l = current_h[index]
                label_index = matrix_rtd_one[index]
                position = positions[index]

                p = np.where(group_temp[-1] == position)[0][0]

                group_temp, new_group = get_1_out(group_temp, p, l)

                ts = (position, 1 - l)
                TS[label_index].append(ts)

                if len(new_group) > 1 and len(new_group[0]) > 1:
                    groups[i+1].append((new_group, label_index))
                    for j in range(len(new_group) - 1):
                        TS[new_group[j, -1]].append(ts)
                # else:
                #     groups[i+1].append(([],next_index))

                if index != len(current_h) - 1:
                    group_temp = np.concatenate((group_temp[:, :p], group_temp[:, (p + 1):]), axis=1)

    return preference, graph, TS, C

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
    matrix_type = "5-10"
    c0_id = 0

    if matrix_type == "12-100":
        ###########       C16
        C16, C16_numH, C16_numX, C16_templates = generate_from_template_C16()
        C16_Sglobal = generate_sigma(C16, C16_numH, C16_numX, 0)
        C16_Smetric = generate_sigma(C16, C16_numH, C16_numX, 1)
        C16, numH, numX, Sg, Sm, templates = C16, C16_numH, C16_numX, C16_Sglobal, C16_Smetric, C16_templates
        CLen = C16.shape[0]
        maxTD = 5

        output_file = "12_100--NEW.txt"
        C = C16
    else:
        C14, CLen, _, _ = generate_Cw14()

        output_file = "5_10--NEW.txt"
        C = C14

    ret = find_local_pref(C, 3, c0_id)
    if ret == "algorithm failed":
        print(ret)
    else:
        pref, graph, TS, revised_C = find_local_pref(C, 3, c0_id)
        for i in range(CLen):
            verify_ts(revised_C, c0_id, i, TS, pref)
        with open(output_file, "w") as f:
            f.write("The concept class:")
            f.write("\n")
            f.write("H,X \t")
            for c in revised_C[-1, :-1]:
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
            for i, preference in enumerate(pref):
                f.write("h{}\t".format(i))
                for p in preference:
                    f.write(str(p)+"\t")
                f.write("\n")

            f.write("Teaching Sequence:\n")
            for i, ts in enumerate(TS):
                f.write("h{}\t".format(i))
                for t in ts:
                    f.write("(x{}, {})\t".format(t[0], t[1]))
                f.write("\n")

        show_graph_with_labels(graph)
