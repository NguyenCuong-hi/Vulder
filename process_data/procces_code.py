# Đường dẫn đến thư mục chứa các file csv
import argparse
import logging
import os

import pandas as pd


# Danh sách các file csv




def merged_header_edges(file_list, count_file):
    header_df = None
    edge_files = [f for f in file_list if 'edges' in f]
    for c in edge_files:
        try:
            name_file_split_header = c.split('_')[-1].split('.')[0]
        except:
            print("File is not correct format ( File Name: " + c + " )")
        if name_file_split_header == 'header':
            temp_header = pd.read_csv(folder_in_csv + c, header=0)

        if name_file_split_header == 'data':
            temp_data = pd.read_csv(folder_in_csv + c, header=None, skiprows=1)
            if header_df is None:
                header_df = temp_data
            else:
                header_df = pd.concat([header_df, temp_data])
    num_cols = temp_data.shape[1]
    dummyDic = {}
    header = list(temp_header.iloc[:])

    for col in range(num_cols):
        dummyDic[header[col]] = list(header_df.iloc[:, col])

    data_csv = pd.DataFrame(dummyDic)
    logging.debug(header_df)

    output_folder = folder_out_csv + 'out_' + str(count_file) + '\\'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_csv.to_csv(output_folder + 'edges.csv', index=True)

    pass


def merged_header_node(file_list, count_file):
    nodes_files = [f for f in file_list if 'nodes' in f]
    name_file_split_header = ''
    node_data = None

    for file in nodes_files:
        try:
            name_file_split_header = file.split('_')[-1].split('.')[0]
        except:
            print("File is not correct format ( File Name: " + file + " )")
            logging.debug('File is not correct format ( File Name: ' + file + ' )')

        if name_file_split_header == 'header':
            temp_header = pd.read_csv(folder_in_csv + file, header=0)
        if name_file_split_header == 'data':
            temp_data = pd.read_csv(folder_in_csv + file)
            if node_data is None:
                node_data = temp_data
            else:
                node_data = pd.concat([node_data, temp_data])

    numcols = temp_data.shape[1]
    dataFrame = {}
    header_df = list(temp_header.iloc[:])

    for col in range(numcols):
        dataFrame[header_df[col]] = list(node_data.iloc[:col])

    data_csv = pd.DataFrame(dataFrame)
    logging.debug(data_csv)

    output_folder = folder_out_csv + 'out_' + str(count_file) + '\\'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_csv.to_csv(output_folder + 'nodes.csv', index=True)
    pass


if __name__ == '__main__':
    paresed = argparse.ArgumentParser()
    paresed.add_argument('--input', type=str, help='Input folfer csv', default='.\\data\\csv\\0_0\\')
    paresed.add_argument('--output', type=str, help='Output file merged header', default='.\\data\\out_header\\')
    paresed.add_argument('--index', type=int, help='Index of folder', default='0')

    arg = paresed.parse_args()
    # folder_in_csv = "D:\\datasets\\data\\csv\\10_0\\"
    # folder_out_csv = "D:\\NCKH\\vuldet\\Vulder\\data\\data_csv"

    folder_in_csv = arg.input
    folder_out_csv = arg.output
    file_index = arg.index

    file_list = os.listdir(folder_in_csv)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, filename='runtime_merge_file.log', filemode='w', format=(
        '%(levelname)s:\t'
        '%(filename)s:'
        '%(lineno)d\t'
        '%(message)s:\t\t'
    )
                        )
    merged_header_edges(file_list, file_index)
    merged_header_node(file_list, file_index)
