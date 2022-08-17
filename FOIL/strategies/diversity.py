import random
import numpy as np
import pandas as pd


def diversity_sampling_strategy_global(classifier, X, n_instances=1):

    # Global Consideration
    centroids = np.random.choice(range(len(X)), size=n_instances, replace=False)
    # print("centroids：", centroids)
    changed, newCentroids = Update_cen(X, centroids, n_instances)
    while changed > 0:
        changed, newCentroids = Update_cen(X, newCentroids, n_instances)

    # centroids = sorted(newCentroids.tolist())
    centroids = newCentroids
    cluster = []
    dis = Distance(X, centroids, n_instances)
    maxIndex = np.argmax(dis, axis=1)
    for i in range(n_instances):
        cluster.append([])
    for i, j in enumerate(maxIndex):
        cluster[j].append(i)

    # return centroids, cluster
    return centroids, np.array(X)[centroids]


def diversity_sampling_strategy_local(classifier, X, n_instances=1):
    select = []
    for i in range(n_instances):
        # need to be changed for more selection method
        # Need to deleted selected sample from X
        select_sam_id = 14
        select_sam = X[select_sam_id]
        add = 0
        for added in select:
            if similarity_sample(X[added], select_sam) > 0.5:
                add = 1
                break
        if not add:
            select.append(select_sam_id)



# Helper
def find_object(sample, index):
    for key in sample['object_detect']['object']:
        if int(key) == index:
            return sample['object_detect']['object'][key]['name']


def compare_tuple(str1, str2, mode):
    # Compare (A,B) with (C,D)
    if mode == 1:
        if str1[:str1.index(',')] == str2[:str2.index(',')] and str1[str1.index(',') + 1:] == str2[str2.index(',') + 1:]:
            return True
        elif str1[:str1.index(',')] == str2[str2.index(',') + 1:] and str1[str1.index(',') + 1:] == str2[:str2.index(',')]:
            return True
        else:
            return False
    if mode == 2:
        # A,B A,B; A,B B,A; A,C A,B B,A; A,C C,B B,C; A,C B,D
        result = 0
        if str1[:str1.index(',')] == str2[:str2.index(',')]:
            result += 0.5
            if str1[str1.index(',') + 1:] == str2[str2.index(',') + 1:]:
                result += 0.5
        elif str1[:str1.index(',')] == str2[str2.index(',') + 1:]:
            result += 0.5
            if str1[str1.index(',') + 1:] == str2[:str2.index(',')]:
                result += 0.5
        elif str1[str1.index(',') + 1:] == str2[:str2.index(',')]:
            result += 0.5
            if str1[:str1.index(',')] == str2[str2.index(',') + 1:]:
                result += 0.5
        elif str1[str1.index(',') + 1:] == str2[str2.index(',') + 1:]:
            result += 0.5
            if str1[:str1.index(',')] == str2[:str2.index(',')]:
                result += 0.5
        return  result


def get_object_lst(sample):
    result = []
    for key in sample['object_detect']['object']:
        obj = sample['object_detect']['object'][key]['name']
        if obj not in result:
            result.append(obj)
    return result


def common_elements_metric(list1, list2, mode):
    if not list1 or not list2:
        return 0
    if mode == 1:
        return len([element for element in list1 if element in list2])/max(len(list1), len(list2))
    if mode == 2:
        acc = 0
        for ele1 in list1:
            for ele2 in list2:
                acc += compare_tuple(ele1[ele1.index('(') + 1:ele1.index(')')], ele2[ele2.index('(') + 1:ele2.index(')')], 2)
        return acc/max(len(list1), len(list2))


def get_overlap_lst(sample):
    result = []
    for key in sample['object_detect']['overlap']:
        overlap_item = 'overlap({},{})'.format(find_object(sample, sample['object_detect']['overlap'][key]['idA']),
                                               find_object(sample, sample['object_detect']['overlap'][key]['idB']))
        is_in = 0
        for item in result:
            if compare_tuple(overlap_item[overlap_item.index('(') + 1:overlap_item.index(')')], item[item.index('(') + 1:item.index(')')], 1):
                is_in = 1
                break
        if not is_in:
            result.append(overlap_item)
    return result


def similarity_sample(s1, s2):
    ob_lst_1 = get_object_lst(s1)
    ob_lst_2 = get_object_lst(s2)
    # print(ob_lst_1)
    # print(ob_lst_2)
    object_sim = common_elements_metric(ob_lst_1, ob_lst_2, 1)

    overlap_lst_1 = get_overlap_lst(s1)
    overlap_lst_2 = get_overlap_lst(s2)
    # print(overlap_lst_1)
    # print(overlap_lst_2)
    overlap_sim = common_elements_metric(overlap_lst_1, overlap_lst_2, 2)
    return object_sim + overlap_sim


def Distance(dataSet, centroids, k) -> np.array:
    dis = []
    for sample in dataSet:
        # if sample not in centroids:
        cent_sim = []
        for cent in centroids:
            cent_sim.append(similarity_sample(sample, dataSet[cent]))
        dis.append(np.array(cent_sim))
    dis = np.array(dis)
    return dis


def Update_cen(dataSet, centroids, k):
    distance = Distance(dataSet, centroids, k)
    # print(distance)
    maxIndex = np.argmax(distance, axis=1)
    # print("len(dataset)", len(dataSet))
    # print("len(maxIndex)", len(maxIndex))
    # print("maxIndex: ", maxIndex)
    cluster = []
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(maxIndex):
        cluster[j].append(i)
    # print("cluster: ", cluster)
    # Find centroid for each cluster
    newCentroids = []
    for i in range(k):
        sam_sum_lst = []
        for sample in cluster[i]:
            sam_sum = 0
            for other_sam in cluster[i]:
                sam_sum += similarity_sample(dataSet[sample], dataSet[other_sam])
            sam_sum_lst.append(sam_sum)
        index_lst = [cluster[i][j] for j, value in enumerate(sam_sum_lst) if value == max(sam_sum_lst)]
        if index_lst:
            max_index = index_lst[0]
        else:
            max_index = []
        newCentroids.append(max_index)
    # print("newCentroids: ", newCentroids)
    # print("oldCentroids: ", centroids)

    # changed = newCentroids - centroids
    changed = 0
    # print(type(centroids))
    for cen in newCentroids:
        if cen not in centroids:
            changed += 1
    # print(changed)

    return changed, newCentroids


