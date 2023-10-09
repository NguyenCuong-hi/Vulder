import argparse, os
import json
import numpy as np


def split_and_save(name, output, buggy, non_buggy, percent, keep_original=False):
    np.random.shuffle(buggy)
    np.random.shuffle(non_buggy)
    num_bug = len(buggy)
    if keep_original:
        num_non_bug = len(non_buggy)
    else:
        num_non_bug = int(num_bug * 100 / percent)
    non_buggy_selected = non_buggy[:num_non_bug]

    train_examples = []
    valid_examples = []
    test_examples = []

    num_train_bugs = int(num_bug * 0.80)
    num_valid_bug = int(num_bug * 0.10)
    train_examples.extend(buggy[:num_train_bugs])
    valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
    test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])

    num_non_bug = len(non_buggy_selected)
    num_train_nobugs = int(num_non_bug * 0.80)
    num_valid_nobug = int(num_non_bug * 0.10)
    train_examples.extend(non_buggy_selected[:num_train_nobugs])
    valid_examples.extend(non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
    test_examples.extend(non_buggy_selected[(num_train_nobugs + num_valid_nobug):])

    final_bug_percentage = int(num_bug * 100 / (num_bug + num_non_bug))
    final_non_bug_percentage = 100 - final_bug_percentage
    file_name = os.path.join(output, name)
    if not keep_original:
        file_name = file_name + '-' + str(final_bug_percentage) + '-' + str(final_non_bug_percentage)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

    for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
        f_name = os.path.join(
            file_name, n + '_GGNNinput.json' )
        print('Saving to, ' + f_name)
        with open(f_name, 'w') as fp:
            json.dump(examples, fp)
            fp.close()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path of the input file', required=True)
    parser.add_argument('--output', help='Output Directory', required=True)
    parser.add_argument('--percent', nargs='+', type=int, help='Percentage of buggy to all', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    input_data = json.load(open(args.input))
    print('Finish Reading data, #examples', len(input_data))
    buggy = []
    non_buggy = []
    for example in input_data:
        target = example['targets'][0][0]
        if target == 1:
            buggy.append(example)
        else:
            non_buggy.append(example)
    print('Buggy', len(buggy), 'Non Buggy', len(non_buggy))
    buggy_count = len(buggy)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    split_and_save(args.name + '-original', args.output, buggy, non_buggy, percent, True)
    for percent in args.percent:
        split_and_save(args.name, args.output, buggy, non_buggy, percent)

    #split_and_save(args.name + '-original', args.output, buggy, non_buggy, percent, True)
    pass
