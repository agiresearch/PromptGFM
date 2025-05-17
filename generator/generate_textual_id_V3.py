import random
import pandas as pd
from collections import defaultdict
import argparse
import openai
import os

openai.api_key = ''


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate textual ID')
    parser.add_argument('--dataset', type=str, default="citeseer", help='Dataset to Process')
    parser.add_argument('--edge_file', type=str, default='graph.csv', help='Directory of the node edge link')
    parser.add_argument('--node_text_arrtibute', type=str, default='node_info.csv',
                        help='Directory of the text attributes of ID')
    parser.add_argument('--generate_model', type=str, default='gpt-3.5-turbo', help='Large language model')
    parser.add_argument('--round', type=int, default=2, help='Generating Round')
    parser.add_argument('--first_path', type=str, default='text_feature.csv', help='First Round Texual ID')
    parser.add_argument('--sample', type=float, default=0.3, help='Sample Ratio')
    return parser.parse_args()


def load_graph_data(graph_data):
    connections = defaultdict(list)
    for index, row in graph_data.iterrows():
        paper_id1, paper_id2 = row[0], row[1]
        connections[paper_id1].append(paper_id2)
        connections[paper_id2].append(paper_id1)
    return connections


def find_1hops(start_node, connections):
    return connections[start_node] if start_node in connections else []


def generate_prompt(node_info, node_id):
    node_info_row = node_info[node_info['paper_id'] == node_id]
    if not node_info_row.empty:
        title = node_info_row['title'].values[0]
        abstract = node_info_row['abs'].values[0]
        prompts_pool = [
            "The title of the paper is {title}, the abstract of the paper is {abstract}. Use about 8 words to summarize the paper.",
            "The title of the paper is {title}, the abstract of the paper is {abstract}. Please summarize the paper in 8 words.",
            "The title of the paper is {title}, the abstract of the paper is {abstract}. Please summarize this article in a single, concise paragraph, within 8 words.",
            "The title of the paper is {title}, the abstract of the paper is {abstract}. Provide a brief summary of this article in one continuous text within 8 words.",
            "The title of the paper is {title}, the abstract of the paper is {abstract}. Summarize this article briefly without listing points within 8 words."
        ]
        selected_prompt = random.choice(prompts_pool)
        return selected_prompt.format(title=title, abstract=abstract)
    else:
        return "Illigal ID"


def generate_answer(prompt, llm, tem):
    response = openai.ChatCompletion.create(
        model=llm,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=tem
    )
    return response['choices'][0]['message']['content']


def generate_first_round_answer(node_info_path, model):
    ids_df = pd.read_csv(node_info_path)
    first_round_answer = ""
    node_textual_id = []
    tem = 0
    node_dict = {}

    for paper in ids_df['paper_id']:
        prompt = generate_prompt(ids_df, paper)
        first_round_answer = generate_answer(prompt, model, tem)
        print(first_round_answer)
        while first_round_answer in node_textual_id:
            first_round_answer = generate_answer(prompt, model, tem)
            if not first_round_answer in node_textual_id:
                tem = 0
            else:
                tem = tem + 0.1

        node_textual_id.append(first_round_answer)
        node_dict[paper] = first_round_answer

    return node_dict


def generate_prompt_iterative(center_desc, one_hop_descs):
    prompt = (
        f"This is the descriptions of the central node: {center_desc}, "
        f"these are the descriptions of the one hop node: {one_hop_descs}, "
        f"please summarize the description of the central node in less than 8 words."
    )
    return prompt


def update_node_features(node_id, node_features, available_connections, used_nodes, sample_ratio, generate_model):
    one_hop_nodes = available_connections[node_id]
    available_nodes = list(one_hop_nodes - used_nodes)
    if len(available_nodes) == 0:
        return node_features[node_id]

    sample_size = min(len(available_nodes), max(1, int(len(available_nodes) * sample_ratio)))
    sampled_nodes = random.sample(available_nodes, sample_size)
    used_nodes.update(sampled_nodes)

    center_desc = node_features[node_id]
    one_hop_descs = ' '.join(node_features[n] for n in sampled_nodes)
    prompt = generate_prompt_iterative(center_desc, one_hop_descs)
    new_description = generate_answer(prompt, generate_model, 0)
    print(new_description)
    return new_description


def iterative_update(node_features, available_connections, rounds, sample_ratio, generate_model, dataset):
    used_nodes = set()
    for r in range(1, rounds + 1):
        new_features = {}
        for node_id in node_features.keys():
            new_features[node_id] = update_node_features(node_id, node_features, available_connections, used_nodes,
                                                         sample_ratio, generate_model)

        # 保存本轮更新结果
        updated_node_info = pd.DataFrame(list(new_features.items()), columns=['ID', 'Description'])
        updated_node_info.to_csv(f'{dataset}_text_feature_round_{r}.csv', index=False)

        # 更新特征字典
        node_features.update(new_features)


def main():
    args = parse_arguments()

  
    root_path = ''

    edge_file_path = os.path.join(root_path, 'dataset', args.dataset, args.edge_file)
    node_info_path = os.path.join(root_path, 'dataset', args.dataset, args.node_text_arrtibute)

    link = pd.read_csv(edge_file_path)
    connections = load_graph_data(link)
    #
    # Generate first round answers
    node_dict = generate_first_round_answer(node_info_path, args.generate_model)
    df = pd.DataFrame(list(node_dict.items()), columns=['ID', 'Description'])
    csv_filename = f"{args.dataset}_{args.first_path}"
    df.to_csv(csv_filename, index=False)

    # node_info = df
    # 迭代特征聚合
    node_info = pd.read_csv(csv_filename)
    node_features = {row['ID']: row['Description'] for _, row in node_info.iterrows()}
    available_connections = {node_id: set(connections[node_id]) for node_id in connections}

    iterative_update(node_features, available_connections, args.round, args.sample, args.generate_model, args.dataset)

if __name__ == "__main__":
    main()
