import subprocess
from spectraljet import FormJets
import numpy as np
import awkward as ak
import os

def compile_fastjet():
    os.chdir("spectraljet")
    output = subprocess.run("g++ applyFastJet.cc -o applyFastJet `/usr/local/bin/fastjet-config --cxxflags --libs --plugins`", 
                            shell=True,
                            capture_output=True)
    os.chdir("..")
    output.check_returncode()
    return output


def run_applyfastjet(eventWise, DeltaR, algorithm_num, jet_name,
                     program_path="./spectraljet/applyFastJet"):
    input_lines = produce_summary(eventWise)
    output_lines = _run_applyfastjet(input_lines, str(DeltaR),
                                     str(algorithm_num), program_path)
    jets = read_fastjet(output_lines, jet_name=jet_name)
    return jets


def produce_summary(eventWise):
    """
    Create a csv of the jet inputs for one event.
    Can be used to sent to other programs or collaborators.

    Parameters
    ----------
    eventWise : EventWise
        file containing data

    Returns
    -------


    """
    assert eventWise.selected_event is not None
    summary = np.vstack((ak.to_numpy(eventWise.JetInputs_SourceIdx),
                         ak.to_numpy(eventWise.JetInputs_Px),
                         ak.to_numpy(eventWise.JetInputs_Py),
                         ak.to_numpy(eventWise.JetInputs_Pz),
                         ak.to_numpy(eventWise.JetInputs_Energy))).T
    summary = summary.astype(str)
    rows = [' '.join(row) for row in summary]
    return '\n'.join(rows).encode()


def _run_applyfastjet(input_lines, DeltaR, algorithm_num, program_path="./spectraljet/applyFastJet", tries=0):
    """
    Run applyfastjet, sending the provided input lines to stdin
    Helper function for run_FastJet

    Parameters
    ----------
    input_lines : list of byte array
        contents of the input as byte arrays
    DeltaR : float
        stopping parameter for clustering
    algorithm_num : int
        number indicating the algorithm to use
    program_path : string
        path to call the program at
        (Default value = "./src/applyFastJet")
    tries : int
        number of tries with this input
        (Default value = 0)


    Returns
    -------
    output_lines : list of bytes
        returned from fastjet

    """
    # input liens should eb one long byte string
    assert isinstance(input_lines, bytes)
    process = subprocess.Popen([program_path, DeltaR, algorithm_num],
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    while process.poll() is None:
        output_lines = None
        process_output = process.stdout.readline()
        if process_output[:2] == b' *':  # all system prompts start with *
            # note that SusHi reads the input file several times
            if b'**send input file to stdin' in process_output:
                process.stdin.write(input_lines)
                process.stdin.flush()
                process.stdin.close()
            elif b'**output file starts here' in process_output:
                process.wait()  # ok let it complete
                output_lines = process.stdout.readlines()
    if output_lines is None:
        print("Error! No output, retrying that input")
        tries += 1
        if tries > 5:
            print("Tried this 5 times... already")
            raise RuntimeError("Subrocess problems")
        # recursive call
        output_lines = _run_applyfastjet(input_lines, DeltaR,
                                         algorithm_num, program_path,
                                         tries)
    return output_lines


def read_fastjet(arg, jet_name="FastJet", do_checks=False):
    """
    Read the outputs of the fastjet program into a PseudoJet

    Parameters
    ----------
    arg : list of strings
        A list of strings it is the byte output of the fastjet program
    jet_name : string
        Name of the jet to be prefixed in the eventWise
        (Default value = "FastJet")
    do_checks : bool
        If checks ont he form of the fastjet output should be done (slow)
        (Default value = False)


    Returns
    -------
    new_pseudojet : PseudoJet
        the peseudojets read from the program

    """
    #  fastjet format
    header = arg[0].decode()[1:]
    arrays = [[]]
    a_type = int
    for line in arg[1:]:
        line = line.decode().strip()
        if line[0] == '#':  # moves from the ints to the doubles
            arrays.append([])
            a_type = float
            fcolumns = line[1:].split()
        else:
            arrays[-1].append([a_type(x) for x in line.split()])
    assert len(arrays) == 2, f"Problem with input; \n{arg}"
    fast_ints = np.array(arrays[0], dtype=int)
    fast_floats = np.array(arrays[1], dtype=float)
    # first line will be the tech specs and columns
    header = header.split()
    DeltaR = float(header[0].split('=')[1])
    algorithm_name = header[1]
    if algorithm_name == 'kt_algorithm':
        ExpofPTInput = 1
    elif algorithm_name == 'cambridge_algorithm':
        ExpofPTInput = 0
    elif algorithm_name == 'antikt_algorithm':
        ExpofPTInput = -1
    else:
        raise ValueError(f"Algorithm {algorithm_name} not recognised")
    # get the colums for the header
    icolumns = {name: i for i, name in
                enumerate(header[header.index("Columns;") + 1:])}
    # and from this get the columns
    # the file of fast_ints contains
    n_fastjet_int_cols = len(icolumns)
    if len(fast_ints.shape) == 1:
        fast_ints = fast_ints.reshape((-1, n_fastjet_int_cols))
    else:
        assert fast_ints.shape[1] == n_fastjet_int_cols
    next_free = np.max(fast_ints[:, icolumns["Label"]], initial=-1) + 1
    fast_idx_dict = {}
    for line_idx, label in fast_ints[:, [icolumns["pseudojet_id"],
                                     icolumns["Label"]]]:
        if label == -1:
            fast_idx_dict[line_idx] = next_free
            next_free += 1
        else:
            fast_idx_dict[line_idx] = label
    fast_idx_dict[-1] = -1
    fast_ints = np.vectorize(fast_idx_dict.__getitem__,
                             otypes=[np.float])(
                                fast_ints[:, [icolumns["pseudojet_id"],
                                              icolumns["parent_id"],
                                              icolumns["child1_id"],
                                              icolumns["child2_id"]]])
    # now the Label is the first one and the pseudojet_id can be removed
    del icolumns["pseudojet_id"]
    icolumns = {name: i-1 for name, i in icolumns.items()}
    n_fastjet_float_cols = len(fcolumns)
    if do_checks:
        # check that the parent child relationship is reflexive
        for line in fast_ints:
            identifier = f"pseudojet inputIdx={line[0]} "
            if line[icolumns["child1_id"]] == -1:
                assert line[icolumns["child2_id"]] == -1, \
                        identifier + "has only one child"
            else:
                assert line[icolumns["child1_id"]] != \
                        line[icolumns["child2_id"]], \
                        identifier + " child1 and child2 are same"
                child1_line = fast_ints[fast_ints[:, icolumns["Label"]]
                                        == line[icolumns["child1_id"]]][0]
                assert child1_line[1] == line[0], \
                    identifier + \
                    " first child dosn't acknowledge parent"
                child2_line = fast_ints[fast_ints[:, icolumns["Label"]]
                                        == line[icolumns["child2_id"]]][0]
                assert child2_line[1] == line[0], \
                    identifier + " second child dosn't acknowledge parent"
            if line[1] != -1:
                assert line[icolumns["Label"]] != \
                    line[icolumns["parent_id"]], \
                    identifier + "is it's own mother"
                parent_line = fast_ints[fast_ints[:, icolumns["Label"]]
                                        == line[icolumns["parent_id"]]][0]
                assert line[0] in parent_line[[icolumns["child1_id"],
                                               icolumns["child2_id"]]], \
                    identifier + " parent doesn't acknowledge child"
        for fcol, expected in zip(fcolumns, FormJets.Clustering.float_columns):
            assert expected.endswith(fcol)
        if len(fast_ints) == 0:
            assert len(fast_floats) == 0, \
                "No ints found, but floats are present!"
            print("Warning, no values from fastjet.")
    if len(fast_floats.shape) == 1:
        fast_floats = fast_floats.reshape((-1, n_fastjet_float_cols))
    else:
        assert fast_floats.shape[1] == n_fastjet_float_cols
    if len(fast_ints.shape) > 1:
        num_rows = fast_ints.shape[0]
        assert len(fast_ints) == len(fast_floats)
    elif len(fast_ints) > 0:
        num_rows = 1
    else:
        num_rows = 0
    ints = np.full((num_rows, len(FormJets.Clustering.int_columns)), -1, dtype=int)
    floats = np.zeros((num_rows, len(FormJets.Clustering.float_columns)), dtype=float)
    if len(fast_ints) > 0:
        ints[:, :4] = fast_ints
        floats[:, :7] = fast_floats
    # make ranks
    rank = -1
    rank_col = len(icolumns)
    ints[ints[:, icolumns["child1_id"]] == -1, rank_col] = rank
    # parents of the lowest rank is the next rank
    this_rank = set(ints[ints[:, icolumns["child1_id"]] == -1,
                    icolumns["parent_id"]])
    this_rank.discard(-1)
    while len(this_rank) > 0:
        rank += 1
        next_rank = []
        for i in this_rank:
            ints[ints[:, icolumns["Label"]] == i, rank_col] = rank
            parent = ints[ints[:, icolumns["Label"]] == i,
                          icolumns["parent_id"]]
            if parent != -1 and parent not in next_rank:
                next_rank.append(parent)
        this_rank = next_rank
    # create the pseudojet
    dict_jet_params = dict(DeltaR=DeltaR,
                           ExpofPTInput=ExpofPTInput,
                           ExpofPTFormatInput='genkt')
    new_pseudojet = FormJets.GeneralisedKT((ints, floats),
                                           jet_name=jet_name,
                                           dict_jet_params=dict_jet_params,
                                           memory_cap=len(ints)+1)
    return new_pseudojet


if __name__ == "__main__":
    from spectraljet import Components
    from matplotlib import pyplot as plt
    eventWise = Components.EventWise.from_file("../megaIgnore/lightHiggs/00_01_signal.parquet")
    eventWise.selected_event = 0
    jets = run_applyfastjet(eventWise, 0.4, 0, "FastJet")
    for jet in jets.split():
        plt.scatter(jet.Leaf_Rapidity, jet.Leaf_Phi)


    
