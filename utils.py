# -*- coding:utf-8 -*-
# Utils For NER
# Author    yanwii 
THRESHOLD = 3

def format_result(result, text, tag): 
    entities = [] 
    for i in result: 
        begin, end = i 
        entities.append({ 
            "begin":begin, 
            "end":end + 1, 
            "entity":text[begin:end+1],
            "type":tag
        }) 
    return {"entities":entities} 

def new_f1_score(tar_path, pre_path, tag, tag_map):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0 if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {}\tprecision {}\tf1 {}".format(tag, recall, precision, f1))
    return recall, precision, f1

def f1_score(tar_path, pre_path, tag, tag_map):
    tp = 0.
    tn = 0.
    fn = 0.
    fp = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)
        for t_tag in tar_tags:
            if t_tag in pre_tags:
                tp += 1
            else:
                fn += 1
        for p_tag in pre_tags:
            if p_tag not in tar_tags:
                fp += 1
    recall = 0 if tp+fn == 0 else (tp/(tp+fn))
    precision = 0 if tp+fp == 0 else (tp/(tp+fp))
    f1 = 0 if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {}\tprecision {}\tf1 {}".format(tag, recall, precision, f1))
    return recall, precision, f1

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S")
    all_tag = [begin_tag, end_tag, single_tag]
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag or tag == single_tag:
            begin = -1
        last_tag = tag
    return tags

def check_begin(self, path, index, o_tag, tags=None):
    if not tags:
        return True
    step = THRESHOLD if index >= THRESHOLD else index
    for i in path[index-step: index]:
        if i == o_tag:
            return False
    return True

def check_end(self, path, index, o_tag, tags=None):
    if not tags:
        return True
    step = (len(path) - index) if (index + THRESHOLD) >= len(path) else THRESHOLD
    for i in path[index: index+step]:
        if i == o_tag:
            return False
    return True


if __name__ == "__main__":
    tag_map = {"B-ORG":1, "I-ORG":2, "E-ORG":3, "O":0}
    f1_score([[0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3]], [[0, 0, 0, 0, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3]], "ORG", tag_map)