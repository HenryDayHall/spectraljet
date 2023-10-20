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


class CALEv2(FormJets.Partitional):
    default_params = {'Sigma': .1,
                      'Cutoff': 0,
                      'WeightExponent': 0.,
                      'SeedGenerator': 'PtCenter',
                      'NRounds': 15}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'WeightExponent': Constants.numeric_classes['pn'],
                       'SeedGenerator': ['PtCenter', 'Random'],
                       'NRounds': Constants.numeric_classes['nn']}

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.SeedGenerator == 'PtCenter':
            self._seed_generator = CALEFunctions.pt_centers
        elif self.SeedGenerator == 'Random':
            self._seed_generator = CALEFunctions.random_centers
        else:
            raise ValueError('SeedGenerator must be PtCenter or Random')

    def setup_internal(self):
        """ Runs before allocate """
        self._seed_labels = []

    def _insert_new_seeds(self):
        """Inserts new seeds into the clustering"""
        energies, pxs, pys, pzs = self._seed_generator(
                self.Avaliable_Energy,
                self.Avaliable_Px, self.Avaliable_Py, self.Avaliable_Pz)
        for energy, px, py, pz in zip(energies, pxs, pys, pzs):
            new_label = np.max(self.Label) + 1
            self._seed_labels.append(new_label)
            new_idx = self._next_free_row()
            # set all relationships to -1
            # and having Parent=-1
            self.Child1[new_idx] = -1
            self.Child2[new_idx] = -1
            self.Parent[jet_idxs] = -1
            self.Rank[new_idx] = -1
            # PT px py pz eta phi energy join_distance
            self.Energy[new_idx] = energy
            self.Px[new_idx] = px
            self.Py[new_idx] = py
            self.Pz[new_idx] = pz
            self.Size[new_idx] = 0.
            # it's easier conceptually to calculate pt, phi and rapidity
            # afresh than derive them
            # for some reason this must be unpacked then assigned
            phi, pt = Components.pxpy_to_phipt(self.Px[new_idx], self.Py[new_idx])
            self.Phi[new_idx] = phi
            self.PT[new_idx] = pt
            self.Rapidity[new_idx] = \
                Components.ptpze_to_rapidity(self.PT[new_idx], self.Pz[new_idx],
                                             self.Energy[new_idx])

    def _remove_seeds(self):
        sorter = np.argsort(self.Label)
        seed_idxs = sorter[np.searchsorted(self.Label, self._seed_labels,
                                           sorter=sorter)]
        # delete all the rows of the float and int arrays containing seeds
        self._ints = np.delete(self._ints, seed_idxs, axis=0)
        self._floats = np.delete(self._floats, seed_idxs, axis=0)
        self._seed_labels.clear()

    def allocate(self):
        """Sort the labels into exclusive jets"""

        available_mask = np.full_like(self.Leaf_Label, True, dtype=bool)
        jet_list = []

        self.laplacien = CALEFunctions.pt_laplacian(
                self.Avaliable_Pt,
                self.Avaliable_Rapidity, self.Avaliable_Phi,
                normalised=self.WeightExponent, sigma=self.Sigma)

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
