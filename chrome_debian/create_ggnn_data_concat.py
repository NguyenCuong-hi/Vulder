import gensim
import os
import argparse
import subprocess
import numpy as np
from gensim.models import Word2Vec
import nltk
import csv
import json

type_map = {
    'CallExpression': 1, 'InclusiveOrExpression': 2, 'ShiftExpression': 3, 'IdentifierDeclStatement': 4,
    'CompoundStatement': 5, 'IdentifierDecl': 6, 'Condition': 7, 'ArgumentList': 8, 'Sizeof': 9,
    'AdditiveExpression': 10, 'BitAndExpression': 11, 'CFGExitNode': 12, 'SizeofOperand': 13, 'AndExpression': 14,
    'Decl': 15, 'Label': 16, 'Argument': 17, 'Function': 18, 'CastExpression': 19, 'IdentifierDeclType': 20,
    'PtrMemberAccess': 21, 'PostIncDecOperationExpression': 22, 'Identifier': 23, 'GotoStatement': 24,
    'UnaryExpression': 25, 'DeclStmt': 26, 'ClassDefStatement': 27, 'FunctionDef': 28, 'File': 29,
    'ParameterType': 30, 'RelationalExpression': 31, 'SwitchStatement': 32, 'ExpressionStatement': 33,
    'CFGEntryNode': 34, 'MultiplicativeExpression': 35, 'Expression': 36, 'ConditionalExpression': 37,
    'MemberAccess': 38, 'ReturnType': 39, 'UnaryOperator': 40, 'BreakStatement': 41, 'CastTarget': 42,
    'ElseStatement': 43, 'SizeofExpression': 44, 'ClassDef': 45, 'DoStatement': 46, 'Symbol': 47,
    'ExclusiveOrExpression': 48, 'Callee': 49, 'ForStatement': 50, 'InitializerList': 51, 'WhileStatement': 52,
    'Statement': 53, 'ContinueStatement': 54, 'PrimaryExpression': 55, 'ParameterList': 56, 'EqualityExpression': 57,
    'Parameter': 58, 'InfiniteForNode': 59, 'IncDec': 60, 'ArrayIndexing': 61, 'CFGErrorNode': 62, 'IfStatement': 63,
    'ForInit': 64, 'UnaryOperationExpression': 65, 'AssignmentExpression': 66, 'ReturnStatement': 67,
    'OrExpression': 68
}

type_one_hot = np.eye(len(type_map))

# We currently consider 12 types of edges mentioned in ICST paper
edgeType = {'IS_AST_PARENT': 1,
            'IS_CLASS_OF': 2,
            'FLOWS_TO': 3,
            'DEF': 4,
            'USE': 5,
            'REACHES': 6,
            'CONTROLS': 7,
            'DECLARES': 8,
            'DOM': 9,
            'POST_DOM': 10,
            'IS_FUNCTION_OF_AST': 11,
            'IS_FUNCTION_OF_CFG': 12}

# edgeType = {'IS_AST_PARENT': 1}


def checkVul(cFile):
    with open(cFile, 'r') as f:
        fileString = f.read()
        return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)


def inputGeneration(nodeCSV, edgeCSV, target, wv, simple=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()

    gInput["targets"].append([target])
    with open(nodeCSV, 'r') as nc:
        # nodes = nc.readlines()
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodes = {}
        node_idx = 0
        for idx, node in enumerate(nodes):
            # node = {'nodeKey': (idx, nodeRpt)}
            # node = set()
            # nodeKey, nodeContent = n.strip().split()[0], n.strip().split()[1:]
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            node_split = nltk.word_tokenize(node_content)
            nrp = np.zeros(128)
            if simple and len(node_split) == 0:
                continue
            max_len = min(len(node_split), 4)
            for tidx in range(max_len):
                token = node_split[tidx]
                try:
                    embedding = wv.wv[token]
                except:
                    embedding = np.zeros(32)
                for idx in range(32):                    
                    nrp[tidx * 32 + idx] = embedding[idx]
            #if len(node_split) > 0:
            #    fNrp = np.divide(nrp, len(node_split))
            #else:
            #    fNrp = nrp
            node_feature = type_one_hot[type_map[node_type] - 1].tolist()
            node_feature.extend(nrp.tolist())
            # gInput["node_features"].append(node_feature)
            allNodes[nodeKey] = node_feature
            nodeMap[nodeKey] = node_idx
            node_idx += 1
        if node_idx == 0 or node_idx >= 500:
            return None

        all_nodes_with_edges = set()
        trueNodeMap = {}
        all_edges = []

        with open(edgeCSV, 'r') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                edge = list()
                start, end, eType = e["start"], e["end"], e["type"]
                if eType == "IS_FILE_OF":
                    # We ignore this for now
                    continue
                else:
                    if not start in nodeMap or not end  in nodeMap or not eType in edgeType:
                        continue
                    edge = [start, edgeType[eType], end]
                    all_edges.append(edge)
                    all_nodes_with_edges.add(start)
                    all_nodes_with_edges.add(end)
                    #gInput["graph"].append(edge)

        dims = []
        for i, node in enumerate(all_nodes_with_edges):
            trueNodeMap[node] = i
            gInput["node_features"].append(allNodes[node])
            dims.append(len(allNodes[node]))

        assert len(set(dims)) == 1, str(set(dims))
        if len(all_edges) == 0:
            return None
        for edge in all_edges:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            gInput["graph"].append(e)

    return gInput



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='normalized csv files to process', default='parsed_results')
    parser.add_argument('--src', help='source c files to process', default='oakland_data')
    parser.add_argument('--simple', action='store_true')
    args = parser.parse_args()

    model = Word2Vec.load('raw_code_deb_chro.32')
    gInputList = list()
    source_files = os.listdir(args.src)
    vul = 0
    non_vul = 0
    i = 0
    done = False
    for i, file_path in enumerate(source_files):
        if i % 100 == 99:
            print(i, vul, non_vul)
        file_path = file_path.strip()
        target = int(file_path[:-2].split('_')[-1])
        nodes_path = os.path.join(args.csv, file_path, 'nodes.csv')
        edges_path = os.path.join(args.csv, file_path, 'edges.csv')
        assert os.path.exists(nodes_path) and os.path.exists(edges_path)
        gInput = inputGeneration(nodes_path, edges_path, target, model, args.simple)
        if gInput is None:
            continue
        if target == 0:
            non_vul += 1
        else:
            vul += 1
        gInputList.append(gInput)
        if vul > 10 and non_vul > 10 and not done:
            with open("oakland_simple_02_11_2020_concat.json", 'w') as f:
                json.dump(gInputList, f)
            done = True
            print("=========== ", vul, non_vul,  "===========")

    if args.simple:
        output_path = "chrome_deb_GGNNinput_02_11_2020_full_concat.json"
    else:
        output_path = "chrome_deb_GGNNinput_02_11_2020_full_concat.json"
    #with open("oakland_simple.json", 'w') as f:
    #    json.dump(gInputList[:10], f)

    with open(output_path, 'w') as gi:
        json.dump(gInputList, gi)
    print(vul, non_vul)
    # cmd = ""
    # # cmd += "cd " + args.csv + ";"
    # cmd += "find " + args.csv + " -name *nodes_normalized.csv;"
    # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
    # result = result.stdout.decode('utf-8')
    # result = result.split("\n")
    # nodesResults = list(filter(None, result))
    #
    # gInputList = list()
    #
    # for nodeCSV in nodesResults:
    #     nodeDir = os.path.dirname(nodeCSV)
    #     edgeCSV = os.path.join(nodeDir, "edges.csv")
    #     # The way to find source file is specific to Joern
    #     sourceFile = os.path.join(args.src, nodeDir.split('/')[-1])
    #     gInput = inputGeneration(sourceFile, nodeCSV, edgeCSV, model)
    #     gInputList.append(gInput)
    #
    # with open(args.csv + "_GGNNinput.json", 'w') as gi:
    #     json.dump(gInputList, gi)


if __name__ == '__main__':
    main()
