import os
import pickle
import torch
import math
import FeatureExtraction
path = 'G:/output_animation/animation_related_API_photo'
dirs = list(range(700))
threshold = 1900
k_VALUE = 40


def find_similar_animation(source_feature, source_dir, k):
    list_k = []
    num = 0
    for dir in dirs:
        path1 = path + '/' + str(dir)
        if not os.path.exists(path1):
            continue
        if dir == source_dir:
            continue
        files = os.listdir(path1)
        for file in files:
            if 'feature.txt' in file:
                num = num + 1
                if num > threshold:
                    return list_k
                target = path1 + '/' + file
                f = open(target, 'rb')
                target_feature = pickle.load(f)
                f.close()
                similarity = torch.dist(source_feature, target_feature, 2)
                similarity = similarity.tolist()
                list_k.append({'similar': similarity, 'path': target})
                new_list_k = sorted(list_k, key=lambda e: e.__getitem__('similar'))
                if len(new_list_k) > k:
                    list_k = new_list_k[0:k-1]
                else:
                    list_k = new_list_k
    print(list_k)
    return list_k


def get_apis(path):
    apis = []
    path1 = path.replace('feature', 'animationApi')
    f = open(path1, encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        api = line.strip()
        if api not in apis:
            apis.append(api)
    return apis


def recommend_API(list_k, top_N, W=0.99):
    API_weight_list = []
    for item in list_k:
        apis = get_apis(item['path'])
        for api in apis:
            flag = True
            for API_weight in API_weight_list:
                if API_weight['API'] == api:
                    weight = math.pow(W, item['similar'])
                    API_weight['weight'] = API_weight['weight'] + weight
                    flag = False
            if flag:
                weight = math.pow(W, item['similar'])
                API_weight_list.append({'API': api, 'weight': weight})
    recommend_API_list = sorted(API_weight_list, key=lambda e: e.__getitem__('weight'), reverse=True)
    returned_API = []
    for api in recommend_API_list[0:top_N]:
        returned_API.append(api['API'])
    return returned_API


def recommend_api(source_path, top_N = 10):
    source_feature = FeatureExtraction.get_feature(source_path)
    # print(source_feature)
    list_k = find_similar_animation(source_feature, 0, k_VALUE)
    Api_list = recommend_API(list_k, top_N)
    return Api_list


if __name__ == '__main__':
    k_VALUE = 50
    top_N = 30
    # source_path = 'G:/output_animation/animation_related_API_photo/131/110.gif'
    source_path = 'C:/Users/wangyihui/Desktop/动画/88.218.gif'
    API_list = recommend_api(source_path, top_N)
    for api in API_list:
        print(api)







