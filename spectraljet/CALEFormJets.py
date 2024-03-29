import numpy as np
from . import FormJets, Constants, CALEFunctions
from .cpp_CALE import build


class CALE(FormJets.Partitional):
    default_params = {'Sigma': .1,
                      'Cutoff': 0,
                      'Normalised': True,
                      'NRounds': 15}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'Normalised': [True, False],
                       'NRounds': Constants.numeric_classes['nn']}

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        pass

    def setup_internal(self):
        """ Runs before allocate """
        self.laplacien, self.l_max_val = CALEFunctions.make_L(
                self.Leaf_Rapidity, self.Leaf_Phi,
                normalised=self.Normalised, sigma=self.Sigma)

    def allocate(self):
        """Sort the labels into exclusive jets"""

        available_mask = np.full_like(self.Leaf_Label, True, dtype=bool)
        jet_list = []

        # Cannot have more jets than input particles.
        max_jets = min(self.NRounds, len(self.Leaf_Rapidity))

        l_idx = CALEFunctions.make_L_idx(self.Leaf_Rapidity, self.Leaf_Phi, self.Leaf_PT)

        # Precompute L_idx sum and its sorted indices
        seed_ordering = l_idx.sum(axis=0).argsort()
        unclustered_idx_pointer = 0 

        round_counter = 0

        while unclustered_idx_pointer < len(seed_ordering) and round_counter < max_jets:
            
            next_unclustered_idx = seed_ordering[unclustered_idx_pointer]

            # If the current seed is not available (already clustered), skip to the next one
            if not available_mask[next_unclustered_idx]:
                unclustered_idx_pointer += 1
                continue
            wavelet_mask = np.zeros_like(self.Leaf_Label, dtype=int)
            wavelet_mask[next_unclustered_idx] = 1

            _, wp_all = CALEFunctions.wavelet_approx(self.laplacien, 2, wavelet_mask)
            wavelet_values = CALEFunctions.min_max_scale(np.array(wp_all[0])).flatten()
            below_cutoff_indices = set(np.where(wavelet_values < self.Cutoff)[0])
            available_particles = set(np.where(available_mask)[0])
            labels = list(below_cutoff_indices & available_particles)

            if labels:  # If we have some labels that match the criteria
                jet_labels = self.Leaf_Label[labels]
                jet_list.append(jet_labels)
                available_mask[labels] = False

            unclustered_idx_pointer += 1
            round_counter += 1

        # Store anything else as a 1-particle jet
        jet_list += [self.Leaf_Label[i:i+1] for i in np.where(available_mask)[0]]

        return jet_list


class CALECpp(FormJets.Partitional):
    default_params = {'Sigma': .1,
                      'Cutoff': 0,
                      'NRounds': 15}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'NRounds': Constants.numeric_classes['nn']}

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        build_dir = Constants.CALE_build_dir
        build.build(build_dir, force_rebuild = False)
        cpp_module = build.get_module(build_dir)
        self.cpp = cpp_module.Cluster(self.Sigma, self.Cutoff, self.NRounds)

    def setup_internal(self):
        """ Runs before allocate """
        self.cpp.SetInputs(self.Leaf_Label, self.Leaf_Energy,
                           self.Leaf_PT, self.Leaf_Rapidity, self.Leaf_Phi)

    def allocate(self):
        """Sort the labels into exclusive jets"""
        self.cpp.DoAllMerges()
        jet_list = self.cpp.GetJetConstituents()
        return jet_list

FormJets.cluster_classes["CALE"] = CALE
FormJets.multiapply_input["CALE"] = CALE
FormJets.cluster_classes["CALECpp"] = CALECpp
FormJets.multiapply_input["CALECpp"] = CALECpp
