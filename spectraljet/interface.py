from spectraljet import FormJets
import sys
import ast
import numpy as np

def get_sequence(jet_class, params, kinematics):
    energy, px, py, pz, pt, rapidity, phi = kinematics.T
    jets = jet_class.from_kinematics(energy, px, py, pz, pt, rapidity, phi,
                                     dict_jet_params=params)
    # the labels that are assigned now are the inputs
    input_labels = jets.Label
    # appart from the -1 placeholders
    input_labels = list(input_labels[input_labels != -1])
    #  the labels that are assigned later  are the clustered labels
    jets.run()
    labels = jets.Label
    mask = labels != -1
    labels = labels[mask]
    child1 = jets.Child1[mask]
    child2 = jets.Child2[mask]
    # Now take advantage of the fact that ids are assigned in order of clustering
    label_order = np.argsort(labels)
    cluster_sequence = []
    for idx in label_order:
        label = labels[idx]
        if label in input_labels:
            continue
        cluster_sequence.append([child1[idx], child2[idx], label])
    remaining = list(jets.Label[mask * jets.Parent == -1])
    return input_labels, cluster_sequence, remaining



def parse_input(input_list):
    jet_class = getattr(FormJets, input_list[0])
    split_at = next(i for i, inp in enumerate(input_list)
                    if inp.endswith("}"))+1
    params = ' '.join(input_list[1:split_at])
    params = ast.literal_eval(params)
    kinematics = np.asarray(input_list[split_at:], dtype=float)
    n_kinematics = 7
    kinematics = kinematics.reshape((-1, n_kinematics))
    return jet_class, params, kinematics


def print_sequence(input_labels, cluster_sequence, remaining):
    text = "input_labels " + str(input_labels) + \
           " cluster_sequence " + str(cluster_sequence) + \
           " remaining " + str(remaining)
    text = text.replace(',', '').replace('[', '[ ').replace(']', ' ]')
    print(text)


def main():
    if len(sys.argv) == 1:
        print("Error, no arguments found")
        sys.exit(1)
    input_args = sys.argv[1:]
    jet_class, params, kinematics = parse_input(input_args)
    if len(kinematics) == 0:
        outputs = [], [], []
    else:
        outputs = get_sequence(jet_class, params, kinematics)
    print_sequence(*outputs)

if __name__ == '__main__':
    main()

