import sys
import os

def main():
    if len(sys.argv) < 9:
        print("Please give train_input file, validation_input_file, test_input file, dict_input, formatted_train_out file,"
              " formatted_validation_out file, formatted_test_out file, and feature_flag respectively in commandline arguments")
    train_input_file = sys.argv[1]
    validation_input_file = sys.argv[2]
    test_input_file = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])

    with open(dict_input, 'r') as f:
        dict_input = f.readlines()
        word_dict = {}
        for line in dict_input:
            key, value = line.split()
            word_dict[key] = value
    with open(train_input_file, 'r') as f:
        train_data = f.readlines()
    with open(validation_input_file, 'r') as f:
        validation_data = f.readlines()
    with open(test_input_file, 'r') as f:
        test_data = f.readlines()

    if feature_flag == 1:
        one_occur(train_data, word_dict, formatted_train_out)
        # one_occur(validation_data, word_dict, formatted_validation_out)
        one_occur(test_data, word_dict, formatted_test_out)

    else:
        one_trim(train_data, word_dict, formatted_train_out)
        # one_trim(validation_data, word_dict, formatted_validation_out)
        one_trim(test_data, word_dict, formatted_test_out)

def one_occur(train_data, word_dict, formatted_train_out):
    # with open(formatted_train_out, 'w') as f:
    #     f.write("")
    for line in train_data:
        label, words = line.split('\t')
        line_dict = {}
        with open(formatted_train_out, 'a') as f:
            f.write(label)
            f.write('\t')
            for word in words.split():
                try:
                    key = word_dict[word]
                    if key not in line_dict.keys():
                        line_dict[word_dict[word]] = 1
                        f.write(key + ":1")
                        f.write('\t')
                except Exception:
                    continue
            f.write('\n')


def one_trim(train_data, word_dict, formatted_train_out):
    # with open(formatted_train_out, 'w') as f:
    #     f.write("")
    for line in train_data:
        label, words = line.split('\t')
        line_dict = {}

        for word in words.split():
            try:
                key = word_dict[word]
                if key in line_dict.keys():
                    line_dict[word_dict[word]] += 1
                else:
                    line_dict[word_dict[word]] = 1
            except Exception:
                continue

        with open(formatted_train_out, 'a') as f:
            f.write(label)
            f.write('\t')
            seen = []
            for word in words.split():
                try:
                    key = word_dict[word]
                    if key not in seen and line_dict[key] < 4:
                        f.write(key + ":1")
                        f.write('\t')
                    seen.append(key)
                except Exception:
                    continue
            f.write('\n')

if __name__ == "__main__":
    main()
