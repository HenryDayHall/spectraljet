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
    _cheby_coeffs = CALEFunctions.cheby_coeff(lambda x: np.exp(-x), 50)
    default_params = {'Sigma': .1,
                      'Cutoff': 0,
                      'WeightExponent': 0.,
                      'SeedGenerator': 'PtCenter'}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'WeightExponent': Constants.numeric_classes['pn'],
                       'SeedGenerator': ['PtCenter', 'Random']}

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
        current_max_label = np.max(self.Label)
        for energy, px, py, pz in zip(energies, pxs, pys, pzs):
            new_label = current_max_label + 1
            current_max_label += 1
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
        # no need to litrally delete the rows,
        # just wipe the label out
        self.Label[seed_idxs] = -1
        self._seed_labels.clear()
        self._avaliable_idxs = [idx for idx in self._avaliable_idxs if idx not in seed_idxs]
        self._avaliable_mask[seed_idxs] = False

    def allocate(self):
        """Sort the labels into exclusive jets"""
        # the assumption is that this function returns a list of
        # constituent labels for each jet
        # then another function is called to actually form the jets
        # As it forms the list it needs to keep track of the avaliable
        # particles itself, but aside from adding and removing seeds,
        # there should eb no modification of the ints and floats
        jet_list = []
        unallocated_leaves = list(self.Leaf_Label)
        # Cannot have more jets than input particles.
        max_jets = min(len(self.Leaf_Rapidity))
        while unallocated_leaves:
            self._insert_new_seeds()
            mask = np.isin(self.Label, unallocated_leaves + self._seed_labels)
            # this will include the current seeds.
            laplacien = CALEFunctions.pt_laplacian(
                    self.Pt[mask], self.Rapidity[mask], self.Phi[mask],
                    weight_exponent=self.WeightExponent, sigma=self.Sigma)
            max_eigval = CALEFunctions.max_eigvalue(laplacien)
            num_points = len(laplacien)
            # going to keep a local avaliable mask for each batch of seeds
            available_mask = np.full(num_points, True, dtype=bool)
            seed_jets = []
            found_content = False
            for seed_label in self._seed_labels:
                wavelet_delta = (self.Label[mask] == seed_label).astype(int)
                # TODO, should the range be from 0 to max_eigval?
                # do the eigenvalues actually go negative?
                wavelets = CALEFunctions.cheby_op(wavelet_delta,
                                                  laplacian, [self._cheby_coeffs],
                                                  (0., max_eigval))
                max_wavelet = np.max(wavelets)
                min_wavelet = np.min(wavelets)
                shifted_cutoff = (max_wavelet - min_wavelet)*(self.Cutoff + 1.)/2. + min_wavelet
                below_cutoff = set(np.where(wavelets < shifted_cutoff)[0])
                avaliable = set(np.where(available_mask)[0])
                jet_content = list(below_cutoff & avaliable)
                if jet_content:
                    found_content = True
                    labels = self.Label[mask][jet_content]
                    jet_list.append(labels)
                    available_mask[jet_content] = False
                    for label in labels:
                        unallocated_leaves.remove(label)
            # this batch of seeds is done.
            self._remove_seeds()
            if not found_content:
                # sometimes the seed leads to no jets
                break  # for now, we just give up if this happens
        if unallocated_leaves:
            # Store anything else as a 1-particle jet
            jet_list += [[label] for label in unallocated_leaves]
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
