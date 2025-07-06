import re

import networkx as nx
import numpy as np
from loguru import logger


def splitCamelCase(word):
    splitted = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", word)).split()
    return splitted


def removeChars(text, chars):
    for c in chars:
        text = text.replace(c, "")
    return text


def get_mde_embedding(text, embedding):
    # >>> get_mde_embedding("WhatDevice", sgram_mde)
    words = splitCamelCase(removeChars(text, ["(", ")", ","]))
    lowercase_list = [s.lower() for s in words]
    counter = 0
    emb = np.zeros(300)
    for w in lowercase_list:
        try:
            emb += embedding[w]
            counter += 1
        except Exception:
            # if failed in emb the complete word, embed the token
            tmp = np.zeros(300)
            for char in list(w):
                
                if char not in embedding:
                    continue
                tmp += embedding[char]
            emb += tmp / len(w)
            counter += 1
    return emb / counter


def cosine_distance(emb_i, emb_j):
    return np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))


def match_classes(raw_1, raw_2, dict_attr, thresh=0.5):
    # raw_1: nodes from reference solution
    # raw_2: nodes from student solution
    # dict_attr: dictionary for embeddings
    # thresh for subst cost
    # map class lists raw_1 and raw_2
    logger.debug(dict_attr)

    def node_subst_cost_attr(node1, node2):
        # threshod as 0.45
        if dict_attr[node1["name"]][node2["name"]] < thresh:
            return 3
        else:
            return 1 - dict_attr[node1["name"]][node2["name"]]

    G_att_1 = nx.Graph()
    G_att_2 = nx.Graph()
    for node in raw_1:
        G_att_1.add_node(node, name=node)

    for node in raw_2:
        G_att_2.add_node(node, name=node)

    for v in nx.optimize_edit_paths(
        G_att_1,
        G_att_2,
        node_subst_cost=node_subst_cost_attr,
        edge_match=None,
        timeout=20,
    ):
        minv = v
    # minv

    return minv


def get_all_info(class_index, class_nodes, list_of_classes, list_edges, need_edge=True):
    result = ""
    result += list_of_classes[class_index] + "\n"
    node = class_nodes[class_index]
    if "abstract" in node:
        node = node.replace("abstract", "").strip()
    if "Abstract" in node:
        node = node.replace("Abstract", "").strip()
    if need_edge:
        for edge in list_edges:
            element = [i.strip() for i in edge.split()]
            if node in element:
                result += edge + "\n"

    return result


def get_embedding(client, text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_cosine_distance(text1, text2, embedding):
    avg = np.sum(embedding[text1], axis=0)
    emb_i = avg / len(text1)

    avg = np.sum(embedding[text2], axis=0)
    emb_j = avg / len(text2)

    return np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))


def create_cosine_similarity_dict(attr_1, attr_2, embedding):
    # attr_1: list[list] split camel case into separated case
    # attr_2: list[list] split camel case into separated case
    # embedding:
    # raw_1: list[str] camcel case attributes
    # raw_2: list[str] camcel case attributes

    # create cosine distance between two attributes with embeeding

    # >>> dic = createCosineDistance(['deviceStatus', 'deviceId'], ['id'], sgram_mde)
    similarities = []
    for att in attr_1:
        emb_i = get_mde_embedding(att, embedding)
        pair = []
        for attribute in attr_2:
            emb_j = get_mde_embedding(attribute, embedding)

            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            pair.append(sim)

        similarities.append(pair)

    dict_attr = {}
    for i in range(len(attr_1)):
        dict_attr[attr_1[i]] = {}
        for j in range(len(attr_2)):
            dict_attr[attr_1[i]][attr_2[j]] = similarities[i][j]

    return dict_attr


def match_attributes(raw_1, raw_2, dict_attr, threshold=0.5):
    # map attributes lists raw_1 and raw_2
    # >>> mapAttributes(['deviceStatus'], ['id'], dic)[0]
    def node_subst_cost_attr(node1, node2):
        # threshod as 0.45
        if dict_attr[node1["name"]][node2["name"]] < threshold:
            return 3
        else:
            return 1 - dict_attr[node1["name"]][node2["name"]]

    G_att_1 = nx.Graph()
    G_att_2 = nx.Graph()
    for node in raw_1:
        G_att_1.add_node(node, name=node)

    for node in raw_2:
        G_att_2.add_node(node, name=node)

    for v in nx.optimize_edit_paths(
        G_att_1,
        G_att_2,
        node_subst_cost=node_subst_cost_attr,
        edge_match=None,
        timeout=20,
    ):
        minv = v
    # minv

    return minv


def check_attributes_type(attr_1, attr_2, ref_dict, stu_dict):
    # given two attribute, compare whether their type can match
    attr_1 = attr_1.split()  # from reference solution do not need to check it
    attr_2 = attr_2.split()

    # if enumerate literal and attributes
    # if len(attr_1) != len(attr_2):
    #   return 0.5, 0.5

    # if an attributes use a regular class as its type
    if len(attr_2) == 2:
        t = attr_2[0].strip()
        cls = stu_dict.get(t, None)
        if cls is not None and (
            stu_dict[t]["type"] == "regular" or stu_dict[t]["type"] == "abstract"
        ):
            return 0.5, 0.5

    if len(attr_1) == 2 or len(attr_2) == 2:
        t = attr_1[0].strip()
        cls = ref_dict.get(t, None)
        # print(cls)
        t_2 = attr_2[0].strip()
        cls_2 = stu_dict.get(t_2, None)
        # print(cls_2)
        if cls is not None:
            if ref_dict[t]["counterpart"] != t_2:
                return 0.5, 0.5
        if cls_2 is not None:
            if stu_dict[t_2]["counterpart"] != t:
                return 0.5, 0.5

    return 1, 1


def create_cosine_similarity_list(attr_1, attr_2, embedding):
    # attr_1: list[list] split camel case into separated case
    # attr_2: list[list] split camel case into separated case
    # embedding:
    # raw_1: list[str] camcel case attributes
    # raw_2: list[str] camcel case attributes

    # create cosine distance between two attributes with embeeding

    # >>> dic = createCosineDistance(['deviceStatus', 'deviceId'], ['id'], sgram_mde)
    similarities = []
    for att in attr_1:
        emb_i = get_mde_embedding(att, embedding)
        pair = []
        for attribute in attr_2:
            emb_j = get_mde_embedding(attribute, embedding)

            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            pair.append(sim)

        similarities.append(pair)

    return similarities


def combine_two_dict(
    list_1, list_2, atr_1, atr_2, cls_1, cls_2, weight_1=0.9, weight_2=0.1
):
    # list_1: similarity between cls
    # list_2: cos similarity between attributes
    if weight_1 + weight_2 != 1:
        raise ValueError("weight_1 + weight_2 != 1")
    result = {}
    for i in range(len(atr_1)):
        atr_r = atr_1[i]
        cls_r = cls_1[i]
        tmp_ref = (atr_r, cls_r)
        result[tmp_ref] = {}
        for j in range(len(atr_2)):
            atr_s = atr_2[j]
            cls_s = cls_2[j]
            tmp_stu = (atr_s, cls_s)

            result[tmp_ref][tmp_stu] = weight_1 * list_1[i][j] + weight_2 * list_2[i][j]

    return result
