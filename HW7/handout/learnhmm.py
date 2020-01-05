import sys
import numpy as np
import matplotlib.pyplot as plt


def process_words_and_tags(index_to_word, index_to_tag):
    with open(index_to_word, 'r') as f:
        index_to_word_data = f.readlines()
    word_to_index_dict = {}
    for i, line in enumerate(index_to_word_data):
        word_to_index_dict[line.strip()] = i
    with open(index_to_tag, 'r') as f:
        index_to_tag_data = f.readlines()
    tag_to_index_dict = {}
    for i, line in enumerate(index_to_tag_data):
        tag_to_index_dict[line.strip()] = i
    return word_to_index_dict, tag_to_index_dict


def process_input(train_input):
    with open(train_input, 'r') as f:
        train_data = f.readlines()
        c = 0
        observations = []
        for data in train_data:
            c += 1
            observations.append(data.strip().split(" "))
            if c == 10000:
                break
    return observations


def get_priors_emits_trans(observations, word_to_index_dict, tag_to_index_dict):
    priors = np.ones(len(tag_to_index_dict))
    emits = np.ones((len(tag_to_index_dict), len(word_to_index_dict)))
    trans = np.ones((len(tag_to_index_dict), len(tag_to_index_dict)))
    for observation in observations:
        for i, word_tag in enumerate(observation):
            word, tag = word_tag.split("_")
            word_id = word_to_index_dict[word]
            tag_id = tag_to_index_dict[tag]
            emits[tag_id][word_id] += 1
            if i == 0:
                priors[tag_id] += 1
            if i < len(observation) - 1:
                word_, tag_ = observation[i+1].split("_")
                trans[tag_id][tag_to_index_dict[tag_]] += 1
    priors /= np.sum(priors)
    emits /= (np.tile(np.sum(emits, axis=1), (emits.shape[1], 1)).transpose())
    trans /= (np.tile(np.sum(trans, axis=1), (trans.shape[1], 1)).transpose())
    return priors, emits, trans


def main():
    if len(sys.argv) < 7:
        print("Please give train_input file, index_to_word file, index_to_tag file, hmm_prior file, "
              "hmm_emit file, and hmm_trans file respectively in commandline arguments")
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmm_prior = sys.argv[4]
    hmm_emit = sys.argv[5]
    hmm_trans = sys.argv[6]

    # x = [10, 100, 1000, 10000]
    # test = [0.8325026284023208, 0.8335539893306335, 0.8564697636384876, 0.9225692145944473]
    # train = [0.8328168509141984, 0.8336861129254841, 0.8607707456500067, 0.9378649549899077]
    # plt.plot(x, train, label='Train Accuracy', marker=".")
    # plt.plot(x, test, label='Test Accuracy', marker=".")
    #
    # plt.xlabel('Number of Sequences')
    # plt.ylabel('Accuracy')
    # plt.title("Accuracy vs Number of Sequences")
    # plt.legend()
    # plt.show()

    observations = process_input(train_input)
    word_to_index_dict, tag_to_index_dict = process_words_and_tags(index_to_word, index_to_tag)
    priors, emits, trans = get_priors_emits_trans(observations, word_to_index_dict, tag_to_index_dict)

    with open(hmm_prior, 'w') as f:
        for prior in priors:
            f.write(str(prior))
            f.write("\n")

    with open(hmm_emit, 'w') as f:
        for emit in emits:
            s = ""
            for v in emit:
                s += (str(v) + " ")
            s = s[:-1] + "\n"
            f.write(s)

    with open(hmm_trans, 'w') as f:
        for tran in trans:
            s = ""
            for v in tran:
                s += (str(v) + " ")
            s = s[:-1] + "\n"
            f.write(s)


if __name__ == "__main__":
    main()
