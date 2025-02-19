from openai import OpenAI
from evaluation.model_metadata import DomainModel
from evaluation.utils import (
    get_mde_embedding,
    cosine_distance,
    match_classes,
    get_embedding,
    get_all_info,
    create_cosine_similarity_dict,
    match_attributes,
    check_attributes_type,
    create_cosine_similarity_list,
    combine_two_dict,
)
from gensim.models import KeyedVectors
import json
import os
from loguru import logger


class Grader:
    def __init__(self, sgram_vectors: KeyedVectors):
        self.sgram_vectors = sgram_vectors

    def calculate_scores(
        self,
        ref_model: DomainModel,
        candidate_model: DomainModel,
        ada_cache_file_loc: str = None,
    ):
        self.match_models(ref_model, candidate_model, ada_cache_file_loc)

        algo_result = {}
        algo_result["class"] = {"precision": 0, "recall": 0, "f1": 0}
        algo_result["attribute"] = {"precision": 0, "recall": 0, "f1": 0}

        # recell
        count = 0
        recall = 0
        for key in list(ref_model.cls_attrs.keys()):
            count += 1
            recall += ref_model.cls_attrs[key]["score"]
        algo_result["class"]["recall"] = recall / count

        # precision
        count = 0
        precision = 0
        for key in list(candidate_model.cls_attrs.keys()):
            count += 1
            precision += candidate_model.cls_attrs[key]["score"]

        algo_result["class"]["precision"] = precision / count
        algo_result["class"]["f1"] = (
            2
            * (algo_result["class"]["recall"] * algo_result["class"]["precision"])
            / (algo_result["class"]["recall"] + algo_result["class"]["precision"])
        )

        # Attributes
        # recell
        # recall
        # count = 0
        # recall = 0
        # for key in list(ref_model.cls_attrs.keys()):
        #     attrs = ref_model.cls_attrs[key]["attributes"]

        # for att in attrs:
        #     count += 1
        #     recall += ref_model.cls_attrs[key]["attributes"][att]["score"]
        # algo_result["attribute"]["recall"] = recall / count

        # # precision
        # count = 0
        # precision = 0
        # for key in list(candidate_model.cls_attrs.keys()):
        #     attrs = candidate_model.cls_attrs[key]["attributes"]

        # for att in attrs:
        #     count += 1
        #     precision += candidate_model.cls_attrs[key]["attributes"][att]["score"]

        # algo_result["attribute"]["precision"] = precision / count

        # r = algo_result["attribute"]["recall"]
        # p = algo_result["attribute"]["precision"]

        # algo_result["attribute"]["f1"] = 2 * (r * p) / (r + p)

        return algo_result

    def match_models(
        self,
        ref_model: DomainModel,
        candidate_model: DomainModel,
        ada_cache_file_loc: str = None,
    ):
        self.match_classes(ref_model, candidate_model, ada_cache_file_loc)
        self.attribute_mapping(ref_model, candidate_model)

    def match_classes(
        self,
        ref_model: DomainModel,
        candidate_model: DomainModel,
        ada_cache_file_loc: str = None,
    ):
        # get name embedding
        # all info
        similarity_mde = []
        mde_embedding = self.sgram_vectors
        threshold = 0.5
        percentage = 0.8
        for index, node in enumerate(ref_model.cls_names):
            # get the class name, remove abstract key word
            cls = node.split()[-1].strip()

            mde_emb_i = get_mde_embedding(cls, mde_embedding)
            mde_emb_i_dsl = get_mde_embedding(
                ref_model.cls_attrs[node]["dsl"], mde_embedding
            )

            mde_pair = []
            for j, stu_node in enumerate(candidate_model.cls_names):
                # get the class name, remove abstract key word
                cls = stu_node.split()[-1].strip()
                mde_emb_j = get_mde_embedding(cls, mde_embedding)
                mde_emb_j_dsl = get_mde_embedding(
                    candidate_model.cls_attrs[stu_node]["dsl"], mde_embedding
                )

                mde_sim = cosine_distance(mde_emb_i, mde_emb_j)
                mde_sim_dsl = cosine_distance(mde_emb_i_dsl, mde_emb_j_dsl)

                mde_pair.append(mde_sim * percentage + (1 - percentage) * mde_sim_dsl)

            # apply third quartile
            similarity_mde.append(mde_pair)

        dict_sim_word = {}
        for i in range(len(ref_model.cls_names)):
            dict_sim_word[ref_model.cls_names[i]] = {}
            for j in range(len(candidate_model.cls_names)):
                dict_sim_word[ref_model.cls_names[i]][candidate_model.cls_names[j]] = (
                    similarity_mde[i][j]
                )

        for key in dict_sim_word:
            exact_match = False
            exact_cls = ""
            for cls_2 in dict_sim_word[key]:
                if dict_sim_word[key][cls_2] > 0.99:
                    exact_match = True
                    exact_cls = cls_2
                    logger.debug(cls_2)
                    break

            if exact_match:  # keep the exact match, others to 0
                logger.debug("update")
                for cls_2 in dict_sim_word[key]:
                    if cls_2 != exact_cls:
                        dict_sim_word[key][cls_2] = 0

        similarity_mde = []
        for i in dict_sim_word:
            tmp = []
            for j in dict_sim_word[i]:
                tmp.append(dict_sim_word[i][j])

            similarity_mde.append(tmp)

        # Stage 1.1.1 Mapping (enum / regular + abstract)
        ref_enum = []
        ref_reg_cls = []
        stu_enum = []
        stu_reg_cls = []

        for enum in ref_model.cls_names:
            if ref_model.cls_attrs[enum]["type"] == "enum":
                ref_enum.append(enum)
            else:
                ref_reg_cls.append(enum)

        for enum in candidate_model.cls_names:
            if candidate_model.cls_attrs[enum]["type"] == "enum":
                stu_enum.append(enum)
            else:
                stu_reg_cls.append(enum)

        enum_mapping = match_classes(
            ref_enum, stu_enum, dict_sim_word, thresh=threshold
        )
        cls_mapping = match_classes(
            ref_reg_cls, stu_reg_cls, dict_sim_word, thresh=threshold
        )

        mapping = enum_mapping[0] + cls_mapping[0]

        for map in mapping:
            if map[0] and map[1]:
                ref_model.cls_attrs[map[0]]["score"] = 1
                ref_model.cls_attrs[map[0]]["counterpart"] = map[1]

                candidate_model.cls_attrs[map[1]]["score"] = 1
                candidate_model.cls_attrs[map[1]]["counterpart"] = map[0]

                if (
                    ref_model.cls_attrs[map[0]]["type"]
                    != candidate_model.cls_attrs[map[1]]["type"]
                ):
                    # type mismatch
                    ref_model.cls_attrs[map[0]]["score"] = 0.5
                    candidate_model.cls_attrs[map[1]]["score"] = 0.5
        # Stage 1.2 Class Mapping (with all info)
        ref_classes = []
        stu_classes = []

        for cls in ref_model.cls_names:
            if ref_model.cls_attrs[cls]["counterpart"] is None:
                ref_classes.append(cls)

        for cls in candidate_model.cls_names:
            if candidate_model.cls_attrs[cls]["counterpart"] is None:
                stu_classes.append(cls)

        # get name embedding
        client = OpenAI()
        # all info
        similarity_all = []
        threshold = 0.7

        if ada_cache_file_loc is not None and os.path.exists(ada_cache_file_loc):
            logger.info(f"Loading cache from {ada_cache_file_loc}")
            with open(ada_cache_file_loc, "r") as json_file:
                dict_sim_all = json.load(json_file)

        else:
            for index, node in enumerate(ref_model.cls_names):
                text_1 = get_all_info(
                    index,
                    ref_model.cls_names,
                    ref_model.cls_raw,
                    ref_model.rel_raw,
                    True,
                )
                emb_i = get_embedding(client, text_1)
                mde_pair = []
                for j, stu_node in enumerate(candidate_model.cls_names):
                    text_2 = get_all_info(
                        j,
                        candidate_model.cls_names,
                        candidate_model.cls_raw,
                        candidate_model.rel_raw,
                        True,
                    )
                    emb_j = get_embedding(client, text_2)
                    mde_sim = cosine_distance(emb_i, emb_j)
                    mde_pair.append(mde_sim)

                # apply third quartile
                similarity_all.append(mde_pair)

            dict_sim_all = {}
            for i in range(len(ref_model.cls_names)):
                dict_sim_all[ref_model.cls_names[i]] = {}
                for j in range(len(candidate_model.cls_names)):
                    dict_sim_all[ref_model.cls_names[i]][
                        candidate_model.cls_names[j]
                    ] = similarity_all[i][j]

        if ada_cache_file_loc is not None and not os.path.exists(ada_cache_file_loc):
            with open(ada_cache_file_loc, "w") as json_file:
                json.dump(dict_sim_all, json_file)

        mapping = match_classes(ref_classes, stu_classes, dict_sim_all, 0.8)
        for map in mapping[0]:
            if map[0] and map[1]:
                ref_model.cls_attrs[map[0]]["score"] = 1
                ref_model.cls_attrs[map[0]]["counterpart"] = map[1]

                candidate_model.cls_attrs[map[1]]["score"] = 1
                candidate_model.cls_attrs[map[1]]["counterpart"] = map[0]

                if (
                    ref_model.cls_attrs[map[0]]["type"]
                    != candidate_model.cls_attrs[map[1]]["type"]
                ):
                    # type mismatch
                    ref_model.cls_attrs[map[0]]["score"] = 0.5
                    candidate_model.cls_attrs[map[1]]["score"] = 0.5

    def attribute_mapping(self, ref_model: DomainModel, candidate_model: DomainModel):
        # map classes
        pairs = []

        # get matched pairs
        for key in ref_model.cls_attrs:
            if ref_model.cls_attrs[key]["counterpart"] is not None:
                pair = [key, ref_model.cls_attrs[key]["counterpart"]]
                pairs.append(pair)

        for pair in pairs:
            if pair[0] and pair[1]:
                logger.debug(pair)

                # map attributes:
                # matched_ref = set()
                # matched_attr = set()

                raw_1 = []
                for attributes in ref_model.cls_attrs[pair[0]]["attributes"]:
                    raw_1.append(attributes)

                raw_2 = []
                for attributes in candidate_model.cls_attrs[pair[1]]["attributes"]:
                    raw_2.append(attributes)

                cos_dict = create_cosine_similarity_dict(
                    raw_1, raw_2, self.sgram_vectors
                )
                mappings = match_attributes(raw_1, raw_2, cos_dict)[0]
                logger.debug(f"mapping {mappings}")

                for mapping in mappings:
                    if mapping[0] is not None and mapping[1] is not None:
                        scores = check_attributes_type(
                            mapping[0],
                            mapping[1],
                            ref_model.cls_attrs,
                            candidate_model.cls_attrs,
                        )
                        ref_model.cls_attrs[pair[0]]["attributes"][mapping[0]][
                            "score"
                        ] = scores[0]
                        ref_model.cls_attrs[pair[0]]["attributes"][mapping[0]][
                            "counterpart"
                        ] = (
                            mapping[1],
                            pair[1],
                        )
                        candidate_model.cls_attrs[pair[1]]["attributes"][mapping[1]][
                            "score"
                        ] = scores[1]
                        candidate_model.cls_attrs[pair[1]]["attributes"][mapping[1]][
                            "counterpart"
                        ] = (
                            mapping[0],
                            pair[0],
                        )

                logger.debug("=" * 20)
        # Stage 2.1.2 Attribute <-> Attribute (between any classes)
        # attr between any classes

        raw_1 = []
        ref_source = []
        tup_r = []
        # get attributes on instrucotr sides
        for cls in ref_model.cls_attrs:
            for attributes in ref_model.cls_attrs[cls]["attributes"]:
                if (
                    ref_model.cls_attrs[cls]["attributes"][attributes]["counterpart"]
                    is None
                ):
                    raw_1.append(attributes)
                    ref_source.append(cls)
                    tup_r.append((attributes, cls))

        raw_2 = []
        stu_source = []
        tup_s = []
        # get attributes on student sides
        for cls in candidate_model.cls_attrs:
            for attributes in candidate_model.cls_attrs[cls]["attributes"]:
                if (
                    candidate_model.cls_attrs[cls]["attributes"][attributes][
                        "counterpart"
                    ]
                    is None
                ):
                    raw_2.append(attributes)
                    stu_source.append(cls)
                    tup_s.append((attributes, cls))

        list_1 = create_cosine_similarity_list(raw_1, raw_2, self.sgram_vectors)
        list_2 = create_cosine_similarity_list(
            ref_source, stu_source, self.sgram_vectors
        )
        combined = combine_two_dict(
            list_1, list_2, raw_1, raw_2, ref_source, stu_source
        )

        mappings = match_attributes(tup_r, tup_s, combined)[0]

        for mapping in mappings:
            logger.debug(mapping)
            if mapping[0] is not None and mapping[1] is not None:
                scores = check_attributes_type(
                    mapping[0][0],
                    mapping[1][0],
                    ref_model.cls_attrs,
                    candidate_model.cls_attrs,
                )

                ref_model.cls_attrs[mapping[0][1]]["attributes"][mapping[0][0]][
                    "score"
                ] = min(scores[0], 0.5)
                ref_model.cls_attrs[mapping[0][1]]["attributes"][mapping[0][0]][
                    "counterpart"
                ] = mapping[1]

                candidate_model.cls_attrs[mapping[1][1]]["attributes"][mapping[1][0]][
                    "score"
                ] = min(scores[1], 0.5)
                candidate_model.cls_attrs[mapping[1][1]]["attributes"][mapping[1][0]][
                    "counterpart"
                ] = mapping[0]

                if ref_model.cls_attrs[mapping[0][1]]["counterpart"] == mapping[1][1]:
                    ref_model.cls_attrs[mapping[0][1]]["attributes"][mapping[0][0]][
                        "score"
                    ] = min(scores[0], 1)

                    candidate_model.cls_attrs[mapping[1][1]]["attributes"][
                        mapping[1][0]
                    ]["score"] = min(scores[1], 1)
        logger.debug("=" * 20)

        # Stage 2.2.1 Attribute mapping atr -> cls
        #  get non-mapped attributes on instrucotr side
        raw_1 = []
        ref_source = []
        tup_r = []
        # get attributes on instrucotr sides
        for cls in ref_model.cls_attrs:
            for attributes in ref_model.cls_attrs[cls]["attributes"]:
                if (
                    ref_model.cls_attrs[cls]["attributes"][attributes]["counterpart"]
                    is None
                ):
                    raw_1.append(attributes)
                    ref_source.append(cls)
                    tup_r.append((attributes, cls))

        raw_2 = []
        stu_source = []
        tup_s = []
        # get un-mapped class on student sides
        for cls in candidate_model.cls_attrs:
            if candidate_model.cls_attrs[cls]["counterpart"] is None:
                raw_2.append(cls)
                stu_source.append(cls)
                tup_s.append((cls, cls))

        list_1 = create_cosine_similarity_list(raw_1, raw_2, self.sgram_vectors)
        list_2 = create_cosine_similarity_list(
            ref_source, stu_source, self.sgram_vectors
        )
        combined = combine_two_dict(
            list_1, list_2, raw_1, raw_2, ref_source, stu_source
        )

        mappings = match_attributes(tup_r, tup_s, combined)[0]

        for mapping in mappings:
            logger.debug(mapping)
            if mapping[0] is not None and mapping[1] is not None:
                ref_model.cls_attrs[mapping[0][1]]["attributes"][mapping[0][0]][
                    "score"
                ] = 0.5
                ref_model.cls_attrs[mapping[0][1]]["attributes"][mapping[0][0]][
                    "counterpart"
                ] = (
                    None,
                    mapping[1][1],
                )

                candidate_model.cls_attrs[mapping[1][1]]["score"] = 0.5
                candidate_model.cls_attrs[mapping[1][1]]["counterpart"] = mapping[0]

        logger.debug("=" * 20)

        # Stage 2.2.2 Attribute mapping cls -> atr
        #  get non-mapped cls on instrucotr side
        raw_1 = []
        ref_source = []
        tup_r = []
        # get class on instrucotr sides
        for cls in ref_model.cls_attrs:
            if ref_model.cls_attrs[cls]["counterpart"] is None:
                raw_1.append(cls)
                ref_source.append(cls)
                tup_r.append((cls, cls))

        raw_2 = []
        stu_source = []
        tup_s = []
        # get un-mapped class on student sides
        for cls in candidate_model.cls_attrs:
            for attributes in candidate_model.cls_attrs[cls]["attributes"]:
                if (
                    candidate_model.cls_attrs[cls]["attributes"][attributes][
                        "counterpart"
                    ]
                    is None
                ):
                    raw_2.append(attributes)
                    stu_source.append(cls)
                    tup_s.append((attributes, cls))

        list_1 = create_cosine_similarity_list(raw_1, raw_2, self.sgram_vectors)
        list_2 = create_cosine_similarity_list(
            ref_source, stu_source, self.sgram_vectors
        )
        combined = combine_two_dict(
            list_1, list_2, raw_1, raw_2, ref_source, stu_source
        )

        mappings = match_attributes(tup_r, tup_s, combined)[0]

        for mapping in mappings:
            logger.debug(mapping)
            if mapping[0] is not None and mapping[1] is not None:
                ref_model.cls_attrs[mapping[0][1]]["score"] = 0.5
                ref_model.cls_attrs[mapping[0][1]]["counterpart"] = mapping[1]

                candidate_model.cls_attrs[mapping[1][1]]["attributes"][mapping[1][0]][
                    "score"
                ] = 0.5
                candidate_model.cls_attrs[mapping[1][1]]["attributes"][mapping[1][0]][
                    "counterpart"
                ] = (
                    None,
                    mapping[0][1],
                )

        logger.debug("=" * 20)
