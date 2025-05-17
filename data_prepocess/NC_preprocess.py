import csv
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser(description='Train the graph language model')
parser.add_argument('--dataset', type=str, default='cora', help='dataset_name')
parser.add_argument('--graph_link', type=str, default='graph.csv', help='Directory of the papers link')
parser.add_argument('--label_path', type=str, default='data.csv', help='Directory of the node ID and label')
parser.add_argument('--generator_outcome', type=str, default='text_feature.csv',
                    help='Directory of the generator_outcome')
parser.add_argument('--outcome', type=str, default='node_classification.csv', help='path of processed dataset, QA format')

args = parser.parse_args()

def load_data(graph_data, label_path, generator_outcome):
    connections = {}
    paper_categories = {}
    paper_texts = {}

    for index, row in graph_data.iterrows():
        paper_id1, paper_id2 = row[0], row[1]
        if paper_id1 not in connections:
            connections[paper_id1] = []
        if paper_id2 not in connections:
            connections[paper_id2] = []
        connections[paper_id1].append(paper_id2)
        connections[paper_id2].append(paper_id1)

    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) == 2:
                paper_id = int(row[0])
                category = row[1]
                paper_categories[paper_id] = category

    with open(generator_outcome, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            paper_id = int(row[0])
            paper_textid = row[1]
            text_id = paper_textid.replace(".", "")
            paper_texts[paper_id] = text_id

    return connections, paper_categories, paper_texts


def find_hops(start_node, connections):
    if start_node not in connections:
        return [], []
    first_hop = connections[start_node]
    second_hop = set()
    for node in first_hop:
        for connected_node in connections.get(node, []):
            if connected_node != start_node and connected_node not in first_hop:
                second_hop.add(connected_node)
    return first_hop, list(second_hop)

def get_category(paper_id, paper_categories):
    return paper_categories.get(paper_id, "Category not found")

def process_node(start_node, connections, paper_categories, paper_texts):
    if int(start_node) not in connections or not connections[int(start_node)]:
        return None  # 如果没有1-hop连接，则不生成数据

    hop_1, hop_2 = find_hops(int(start_node), connections)
    print("hop_1:", hop_1)

    if len(hop_1) > max_hops_1:
        hop_1 = random.sample(hop_1, max_hops_1)
    if len(hop_2) > max_hops_2:
        hop_2 = random.sample(hop_2, max_hops_2)

    start_node_domain = get_category(int(start_node), paper_categories)
    start_node_text_id = paper_texts[int(start_node)]
    input_text = ''
    prompt = f"Which category should <{start_node_text_id}> be classified as?"
    if hop_1:
        input_text = f'<{start_node_text_id}> has one hop connections with: '
        for index, node in enumerate(hop_1):
            node_text = paper_texts[int(node)]
            input_text += f'<{node_text}>' + (', ' if index != len(hop_1) - 1 else '. ')

    if hop_2:
        input_text += f' <{start_node_text_id}> also has two hop connection with: '
        for index, node in enumerate(hop_2):
            node_text = paper_texts[int(node)]
            input_text += f'<{node_text}>' + (', ' if index != len(hop_2) - 1 else '. ')

    return {'input_text': input_text + prompt, 'output_text': start_node_domain}

if __name__ == '__main__':
    # 修改Root Path
    root_path = ''

    graph_link = os.path.join(root_path, "dataset", args.dataset, args.graph_link)
    label_path = os.path.join(root_path, "dataset", args.dataset, args.label_path)
    temp_path = args.dataset + '_' + args.generator_outcome
    generator_outcome = os.path.join(root_path, "generator", temp_path)

    graph = pd.read_csv(graph_link)
    connections, paper_categories, paper_texts = load_data(graph, label_path, generator_outcome)
    print(paper_categories)
    max_hops_1 = 12
    max_hops_2 = 5


    # print("\nTypes in connections dictionary:")
    # for k, v in connections.items():
    #     print(f"Key: {k} -> {type(k)}, Value: {v} -> {type(v)}")


    # print("Find hops")
    # print(find_hops(int(35),connections))

    fieldnames = ['input_text', 'output_text']
    output_path = args.dataset + '_' + args.outcome
    with open(output_path, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        with open(generator_outcome, mode='r', encoding='UTF-8') as path_file:
            path_reader = csv.DictReader(path_file)
            for row in path_reader:
                start_node = row['ID']
                result = process_node(start_node, connections, paper_categories, paper_texts)
                if result:
                    writer.writerow(result)

    dataset = pd.read_csv(output_path)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    train_output_path = 'train_' + args.dataset + '_' + args.outcome
    test_output_path = 'test_' + args.dataset + '_' + args.outcome

    train.to_csv(train_output_path, index=False, encoding='utf-8')
    test.to_csv(test_output_path, index=False, encoding='utf-8')


