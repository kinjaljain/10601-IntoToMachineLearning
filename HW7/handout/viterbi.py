import sys
import numpy as np
import math


def process_input(test_input):
    with open(test_input, 'r') as f:
        train_data = f.readlines()
        observations = []
        tags = []
        for data in train_data:
            words = data.strip().split(" ")
            observation = []
            tag = []
            for word in words:
                observation_, tag_ = word.split("_")
                observation.append(observation_)
                tag.append(tag_)
            observations.append(observation)
            tags.append(tag)
    return observations, tags


def process_words_and_tags(index_to_word, index_to_tag):
    word_to_index_dict = {}
    with open(index_to_word, 'r') as f:
        index_to_word = f.readlines()
        for i, line in enumerate(index_to_word):
            word_to_index_dict[line.strip()] = i
    tag_to_index_dict = {}
    with open(index_to_tag, 'r') as f:
        index_to_tag = f.readlines()
        for i, line in enumerate(index_to_tag):
            tag_to_index_dict[line.strip()] = i
    return word_to_index_dict, tag_to_index_dict


def fetch_priors_emits_trans(hmm_priors, hmm_emits, hmm_trans):
    with open(hmm_priors, 'r') as f:
        priors = []
        hmm_priors = f.readlines()
        for hmm_prior in hmm_priors:
            priors.append(float(hmm_prior.strip()))
    with open(hmm_emits, 'r') as f:
        emits = []
        hmm_emits = f.readlines()
        for hmm_emit in hmm_emits:
            emit = []
            for observation in hmm_emit.strip().split(" "):
                emit.append(float(observation))
            emits.append(np.array(emit))
    with open(hmm_trans, 'r') as f:
        trans = []
        hmm_trans = f.readlines()
        for hmm_tran in hmm_trans:
            tran = []
            for state in hmm_tran.strip().split(" "):
                tran.append(float(state))
            trans.append(np.array(tran))
    return np.array(priors), np.array(emits), np.array(trans)


def viterbi(observations, tags, priors, emits, trans, word_to_index_dict, tag_to_index_dict, index_to_tag_dict):
    predictions = []
    count = 0
    total = 0
    for observation, observation_tag in zip(observations, tags):
        prediction = ""
        predicted_tag = []
        path_probabilities = np.zeros((len(tag_to_index_dict), len(observation)))
        back_pointers = np.zeros((len(tag_to_index_dict), len(observation)), int)

        for tag in range(0, len(tag_to_index_dict)):
            path_probabilities[tag][0] = math.log(priors[tag]) + \
                                         math.log(emits[tag][word_to_index_dict[observation[0]]])
            back_pointers[tag][0] = tag

        for o in range(1, len(observation)):
            for tag in range(0, len(tag_to_index_dict)):
                max_v = - math.inf
                back_pointers[tag][o] = -1
                for tag_ in range(0, len(tag_to_index_dict)):
                    v = path_probabilities[tag_][o-1] + math.log(trans[tag_][tag]) + \
                        math.log(emits[tag][word_to_index_dict[observation[o]]])
                    if v > max_v:
                        max_v = v
                        back_pointers[tag][o] = tag_
                path_probabilities[tag][o] = max_v

        print(path_probabilities)

        tag_index = np.argmax(path_probabilities.transpose()[len(observation)-1])
        predicted_tag.append(index_to_tag_dict[tag_index])
        for i in reversed(range(1, len(observation))):
            tag_index = back_pointers[tag_index][i]
            predicted_tag.append(index_to_tag_dict[tag_index])
        predicted_tag.reverse()
        for tag, predicted in zip(observation_tag, predicted_tag):
            if tag == predicted:
                count += 1
            total += 1
        for o, t in zip(observation, predicted_tag):
            prediction += "_".join([o, t])
            prediction += " "
        prediction = prediction[:-1] + "\n"
        predictions.append(prediction)

    accuracy = float(count/total)
    return predictions, accuracy


def main():
    if len(sys.argv) < 9:
        print("Please give test_input file, index_to_word file, index_to_tag file, hmm_prior file, "
              "hmm_emit file, hmm_trans file, predicted_file, and metric_file respectively in commandline arguments.")
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmm_prior = sys.argv[4]
    hmm_emits = sys.argv[5]
    hmm_trans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    observations, tags = process_input(test_input)
    word_to_index_dict, tag_to_index_dict = process_words_and_tags(index_to_word, index_to_tag)
    index_to_tag_dict = {v: k for k, v in tag_to_index_dict.items()}
    priors, emits, trans = fetch_priors_emits_trans(hmm_prior, hmm_emits, hmm_trans)
    predictions, accuracy = viterbi(observations, tags, priors, emits, trans, word_to_index_dict, tag_to_index_dict,
                                    index_to_tag_dict)

    with open(predicted_file, 'w') as f:
        for prediction in predictions:
            f.write(prediction)

    with open(metric_file, 'w') as f:
        f.write("Accuracy: ")
        f.write(str(accuracy))
        f.write("\n")


if __name__ == "__main__":
    main()
