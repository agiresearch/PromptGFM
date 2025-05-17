import csv
import random
from concurrent.futures import ThreadPoolExecutor
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
parser.add_argument('--outcome', type=str, default='link_prediction.csv', help='path of processed dataset, QA format')


args = parser.parse_args()


def split_citation_data(file_path, test_size=0.3, random_state=42):
    cite_data = pd.read_csv(file_path)
    unique_papers = pd.concat([cite_data['paper_id1'], cite_data['paper_id2']]).unique()
    train_papers, test_papers = train_test_split(unique_papers, test_size=test_size, random_state=random_state)
    train_mask = (cite_data['paper_id1'].isin(train_papers)) & (cite_data['paper_id2'].isin(train_papers))
    test_mask = (cite_data['paper_id1'].isin(test_papers)) & (cite_data['paper_id2'].isin(test_papers))
    train_cite_data = cite_data[train_mask]
    test_cite_data = cite_data[test_mask]
    return train_cite_data, test_cite_data


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


def process_node(start_node, connections, paper_categories, paper_texts):
    hop_1, hop_2 = find_hops(start_node, connections)
    if not hop_1:
        return None  # 如果没有1-hop连接，则不生成数据
    start_node_text_id = paper_texts[int(start_node)]
    masked_node = random.choice(hop_1)
    hop_1.remove(masked_node)
    masked_node_text_id = paper_texts[int(masked_node)]
    other_nodes = set(paper_texts.keys()) - set(hop_1) - set(hop_2) - {start_node}
    random_node = random.choice(list(other_nodes))
    random_node_text_id = paper_texts[random_node]
    prompt = "Which node will be connected to"
    input_text = f'<{start_node_text_id}> has one hop connections with: '
    if hop_1:
        for index, node in enumerate(hop_1):
            node_text = paper_texts[int(node)]
            input_text += f'<{node_text}>' + (', ' if index != len(hop_1) - 1 else '. ')
    if hop_2:
        input_text += f'<{start_node_text_id}> also has two hop connections with: '
        for index, node in enumerate(hop_2):
            node_text = paper_texts[int(node)]
            input_text += f'<{node_text}>' + (', ' if index != len(hop_2) - 1 else '. ')
    input_text += "Among "
    node_options = [masked_node_text_id] + [random_node_text_id]
    random.shuffle(node_options)
    input_text += ", ".join(f'<{node_text}>' for node_text in node_options)
    input_text += f", {prompt} <{start_node_text_id}>?"
    return {'input_text': input_text, 'output_text': '<' + masked_node_text_id + '>'}


def get_category(paper_id, paper_categories):
    return paper_categories.get(paper_id, "Category not found")


if __name__ == '__main__':
    # 修改Root Path
    root_path = ''

    graph_link = os.path.join(root_path, "dataset", args.dataset, args.graph_link)
    label_path = os.path.join(root_path, "dataset", args.dataset, args.label_path)
    temp_path = args.dataset + '_' + args.generator_outcome
    generator_outcome = os.path.join(root_path, "generator", temp_path)

    print(graph_link)
    dataset = graph_link
    train_cite_data, test_cite_data = split_citation_data(dataset, test_size=0.3, random_state=42)

    print("Training citations: ", len(train_cite_data))
    print("Testing citations: ", len(test_cite_data))

    train_connections, train_paper_categories, train_paper_texts = load_data(train_cite_data, label_path, generator_outcome)
    test_connections, test_paper_categories, test_paper_texts = load_data(test_cite_data, label_path, generator_outcome)


    def generate_QA_pairs(connections, paper_categories, paper_texts, output_file):
        fieldnames = ['input_text', 'output_text']
        with open(output_file, mode="w", newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for start_node in connections.keys():
                result = process_node(start_node, connections, paper_categories, paper_texts)
                if result:
                    writer.writerow(result)


    generate_QA_pairs(train_connections, train_paper_categories, train_paper_texts, "train_" + args.dataset+'_' + args.outcome)
    generate_QA_pairs(test_connections, test_paper_categories, test_paper_texts, "test_" + args.dataset+'_' + args.outcome)
