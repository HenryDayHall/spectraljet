from matplotlib import pyplot as plt
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
        self._ca_distances2 = FormJets.ca_distances2(
                self.Leaf_Rapidity, self.Leaf_Phi)
        self.laplacien, self.l_max_val = CALEFunctions.make_L(
                self._ca_distances2,
                normalised=self.Normalised, sigma=self.Sigma)

    def allocate(self):
        """Sort the labels into exclusive jets"""

        available_mask = np.full_like(self.Leaf_Label, True, dtype=bool)
        jet_list = []

        # Cannot have more jets than input particles.
        max_jets = min(self.NRounds, len(self.Leaf_Rapidity))

        l_idx = CALEFunctions.make_L_idx(self._ca_distances2, self.Leaf_PT)

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
    default_params = {'Sigma': 1.,
                      'InvPower': 1.,
                      'Cutoff': 0.3,
                      'WeightExponent': 0.,
                      'SeedGenerator': 'PtCenter',
                      'ToAffinity': 'exp',
                      'SeedIncrement': 3}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'InvPower': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'WeightExponent': [0., Constants.numeric_classes['pdn']],
                       'SeedGenerator': ['PtCenter', 'Random', 'Unsafe'],
                       'ToAffinity': ['exp', 'inv'],
                       'SeedIncrement': Constants.numeric_classes['nn']}
    max_seeds = 30

    def create_int_float_tables(self, start_ints, start_floats):
        """ Format the data for clustering, allocating memory.
        The tables have space for a center point for each pottential cluster.

        Parameters
        ----------
        start_ints : list of list of int
            initial integer input data for clustering
        start_floats : list of list of floats
            initial float input data for clustering

        Returns
        -------
        ints : list of list of int
            integer input data for clustering
        floats : list of list of floats
            float input data for clustering
        """
        ints, floats = super().create_int_float_tables(start_ints, start_floats)
        seed_ints = -np.ones((self.max_seeds, len(self.int_columns)),
                             dtype=int)
        ints = np.vstack((ints, seed_ints))
        seed_floats = -np.ones((self.max_seeds, len(self.float_columns)),
                               dtype=float)
        floats = np.vstack((floats, seed_floats))
        return ints, floats

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.SeedGenerator == 'PtCenter':
            self._seed_generator = CALEFunctions.pt_centers
        elif self.SeedGenerator == 'Random':
            self._seed_generator = CALEFunctions.random_centers
        elif self.SeedGenerator == 'Unsafe':
            self._seed_generator = CALEFunctions.unsafe_centers
        else:
            raise ValueError('SeedGenerator must be PtCenter, Unsafe or Random')
        self._calc_phys_distances2 = FormJets.ca_distances2
        if self.ToAffinity == 'exp':
            def affinity(distances2, sigma=self.Sigma):
                return np.exp(-(distances2)/(2*sigma**2))
        elif self.ToAffinity == 'inv':
            inv_multiplier = -self.InvPower*0.5
            def affinity(distances2, mult=inv_multiplier):
                return distances2**inv_multiplier
        else:
            raise ValueError("ToAffinity must be inv or exp")
        self._calc_affinity = affinity
        def laplacian(pts, distances2, affinities):
            return CALEFunctions.pt_laplacian(pts, distances2, affinities,
                weight_exponent=self.WeightExponent)
        self._calc_laplacian = laplacian

    def setup_internal(self):
        """Setup needed for a particular clustering calculation.
        Runs before allocate.

        Should generate things like matrices of distances.
        """
        # Keep track of indices currently holding seed particles
        self._seed_labels = []
        # in here we will create various permanant matrices
        self.setup_internal_local()
        # in here, matricies that are transient and must be recreated each
        # time the particles change are created
        # self.setup_internal_global()
        # they will be updated when the first seed it added.

    def setup_internal_local(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Only includes local object, where a particle combining
        only updates a small subset of entries, rather than the whole table.
        """
        space_size = (self._ints.shape[0], self._ints.shape[0])
        # distance in physical space
        angular_distances2 = self._calc_phys_distances2(
            self.Available_Rapidity, self.Available_Phi)
        np.fill_diagonal(angular_distances2, 0.)
        self._phys_distances2 = np.empty(space_size, dtype=float)
        self._phys_distances2[self._2d_available_indices] = \
            angular_distances2
        # affinities in physical space
        self._affinity = np.empty(space_size, dtype=float)
        affinity = self._calc_affinity(angular_distances2)
        self._affinity[self._2d_available_indices] = affinity
        # symmetric size, always symmetric in the first step
        self._floats[self._available_idxs, self._col_num["Size"]] =\
            np.sum(affinity, axis=1)

    def _update_matrices(self, new_idxs):
        """Peform updates to internal data, fixing the row of new_idx

        Parameters
        ----------
        new_idxs : list of int
            indices of points needing update
        """
        # physical distances
        new_rapidity = self.Rapidity[new_idxs].reshape((-1, 1))
        new_phi = self.Phi[new_idxs].reshape((-1, 1))
        new_angular_distance2 = self._calc_phys_distances2(
            self.Available_Rapidity, self.Available_Phi,
            new_rapidity, new_phi)
        masked_idxs = [self._available_idxs.index(i) for i in new_idxs]
        mask_2d = self._2d_available_indices
        self._phys_distances2[mask_2d][masked_idxs] = \
            new_angular_distance2
        self._phys_distances2.T[mask_2d][masked_idxs] = \
            new_angular_distance2
        np.fill_diagonal(self._phys_distances2, 0.)

        # new affinity
        new_affinity = self._calc_affinity(new_angular_distance2)
        self._affinity[mask_2d][masked_idxs] = new_affinity
        self._affinity.T[mask_2d][masked_idxs] = new_affinity
        np.fill_diagonal(self._affinity, 0.)

        # everything else needs global calculation
        self.setup_internal_global()

    def setup_internal_global(self):
        """Setup needed for a particular clustering calculation.

        Should generate things like matrices of distances.
        Only includes global calculation where particles combining
        updates all entries, rather than a subset
        As such, this is called after each combination.
        """
        # laplacian
        mask_2d = self._2d_available_indices
        affinity = self._affinity[mask_2d]
        distance2 = self._phys_distances2[mask_2d]
        laplacian = self._calc_laplacian(self.Available_PT, distance2, affinity)
        self._laplacian = laplacian

    def _insert_new_seeds(self, n_seeds, mask):
        """
        Inserts new seeds into the clustering
        """
        seed_pxpypz, seed_ptrapphi = self._seed_generator(
                self.Energy[mask],
                self.Px[mask], self.Py[mask], self.Pz[mask],
                self.PT[mask], self.Rapidity[mask], self.Phi[mask],
                n_centers=n_seeds)
        first_label = np.max(self.Label) + 1
        labels = list(range(first_label, first_label + n_seeds))
        self._seed_labels += labels
        # _next_free_row depends on the label being set
        rows = []
        for label in labels:
            row = self._next_free_row()
            rows.append(row)
            self.Label[row] = label
        # set all relationships to -1
        # and having Parent=-1
        self.Child1[rows] = -1
        self.Child2[rows] = -1
        self.Parent[rows] = -1
        self.Rank[rows] = -1
        # PT px py pz eta phi energy join_distance
        self.Energy[rows] = np.mean(self.Leaf_Energy)
        self.Px[rows] = seed_pxpypz[:, 0]
        self.Py[rows] = seed_pxpypz[:, 1]
        self.Pz[rows] = seed_pxpypz[:, 2]
        self.Size[rows] = 0.
        # it's easier conceptually to calculate pt, phi and rapidity
        # afresh than derive them
        # for some reason this must be unpacked then assigned
        self.Phi[rows] = seed_ptrapphi[:, 2]
        self.PT[rows] = seed_ptrapphi[:, 0]
        self.Rapidity[rows] = seed_ptrapphi[:, 1]
        self._update_avalible([], idxs_in=set(rows))
        self._update_matrices(rows)

    def _remove_seeds(self):
        sorter = np.argsort(self.Label)
        seed_idxs = sorter[np.searchsorted(self.Label, self._seed_labels,
                                           sorter=sorter)]
        # no need to litrally delete the rows,
        # just wipe the label out
        self.Label[seed_idxs] = -1
        self._seed_labels.clear()
        self._update_avalible(seed_idxs)

    def allocate(self, fig=None, ax=None):
        """Sort the labels into exclusive jets"""
        # the assumption is that this function returns a list of
        # constituent labels for each jet
        # then another function is called to actually form the jets
        # As it forms the list it needs to keep track of the available
        # particles itself, but aside from adding and removing seeds,
        # there should eb no modification of the ints and floats
        jet_list = []
        unallocated_leaves = set(self.Leaf_Label)
        n_seeds = self.SeedIncrement
        # Cannot have more jets than input particles.
        while unallocated_leaves:
            mask = np.isin(self.Label, list(unallocated_leaves))
            self._insert_new_seeds(n_seeds, mask)
            mask = np.isin(self.Label, list(unallocated_leaves) + self._seed_labels)
            # this will include the current seeds.
            max_eigval = CALEFunctions.max_eigenvalue(self._laplacian)
            num_points = len(self._laplacian)
            seed_jets = []
            found_content = False
            for seed_label in self._seed_labels:
                seed_location = self.Label[mask] == seed_label
                # TODO, should the range be from 0 to max_eigval?
                # do the eigenvalues actually go negative?
                try:
                    wavelets = CALEFunctions.cheby_op(seed_location.astype(int),
                                                      self._laplacian, [self._cheby_coeffs],
                                                      (-max_eigval, max_eigval))[0]
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    wavelets = CALEFunctions.cheby_op(seed_location.astype(int),
                                                      self._laplacian, [self._cheby_coeffs],
                                                      (-max_eigval, max_eigval))[0]

                max_wavelet = np.max(wavelets[~seed_location])
                min_wavelet = np.min(wavelets[~seed_location])
                shifted_cutoff = (max_wavelet - min_wavelet)*self.Cutoff + min_wavelet
                below_cutoff = set(self.Label[mask][wavelets > shifted_cutoff])
                jet_labels = below_cutoff.intersection(unallocated_leaves)
                if len(jet_labels) > 1:  # if there is only one object, it's not a jet.
                    found_content = True
                    jet_list.append(list(jet_labels))
                    unallocated_leaves -= jet_labels
                if len(unallocated_leaves) < 2:
                    break  # one unallocated leaf cant be a jet
            # this batch of seeds is done.
            self._remove_seeds()
            if not found_content:
                n_seeds += self.SeedIncrement
                # sometimes the seed leads to no jets
                if n_seeds > self.max_seeds:
                    break
        if unallocated_leaves:
            # Store anything else as a 1-particle jet
            jet_list += [[label] for label in unallocated_leaves]
        return jet_list

    def plot_allocate(self, break_after_first=False, ignore_seed_wavelet=False):
        """Sort the labels into exclusive jets"""
        # the assumption is that this function returns a list of
        # constituent labels for each jet
        # then another function is called to actually form the jets
        # As it forms the list it needs to keep track of the available
        # particles itself, but aside from adding and removing seeds,
        # there should eb no modification of the ints and floats
        jet_list = []
        # plotting data ~~~~~~~~~~~
        made_jet = []
        seed_values = []
        stalk_plot_vars = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        unallocated_leaves = set(self.Leaf_Label)
        n_seeds = self.SeedIncrement
        # Cannot have more jets than input particles.
        while unallocated_leaves:
            mask = np.isin(self.Label, list(unallocated_leaves))
            self._insert_new_seeds(n_seeds, mask)
            mask = np.isin(self.Label, list(unallocated_leaves) + self._seed_labels)
            # this will include the current seeds.
            max_eigval = CALEFunctions.max_eigenvalue(self._laplacian)
            num_points = len(self._laplacian)
            seed_jets = []
            found_content = False
            for seed_label in self._seed_labels:
                seed_values.append(seed_label)  # plotting
                seed_location = self.Label[mask] == seed_label
                # TODO, should the range be from 0 to max_eigval?
                # do the eigenvalues actually go negative?
                wavelets = CALEFunctions.cheby_op(seed_location.astype(int),
                                                  laplacien, [self._cheby_coeffs],
                                                  (-max_eigval, max_eigval))[0]
                if ignore_seed_wavelet:  # plotting
                    max_wavelet = np.max(wavelets[~seed_location])
                    min_wavelet = np.min(wavelets[~seed_location])
                else:
                    max_wavelet = np.max(wavelets)
                    min_wavelet = np.min(wavelets)
                shifted_cutoff = (max_wavelet - min_wavelet)*self.Cutoff + min_wavelet
                below_cutoff = set(self.Label[mask][wavelets > shifted_cutoff])
                jet_labels = below_cutoff.intersection(unallocated_leaves)
                if len(jet_labels) > 1:  # if there is only one object, it's not a jet.
                    made_jet.append(True)  # plotting
                    found_content = True
                    jet_list.append(list(jet_labels))
                    unallocated_leaves -= jet_labels
                else:
                    made_jet.append(False)  # plotting
                # Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                jet_content = np.isin(self.Label[mask], list(jet_labels))
                stalk_plot_vars.append((self.Rapidity[mask], self.Phi[mask], wavelets,
                                        jet_content, np.where(self.Label[mask] == seed_label)[0]))
                print(f"max_wavelet, {max_wavelet}, min_wavelet, {min_wavelet}, shifted_cutoff, {shifted_cutoff}")
                print(f"jet length = {sum(jet_content)}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if len(unallocated_leaves) < 2:
                    break  # one unallocated leaf cant be a jet
            # this batch of seeds is done.
            self._remove_seeds()
            # plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if break_after_first:
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not found_content:
                n_seeds += self.SeedIncrement
                # sometimes the seed leads to no jets
                if n_seeds > self.max_seeds:
                    break
        # plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt_clustering(self, seed_values, made_jet, jet_list, stalk_plot_vars)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if unallocated_leaves:
            # Store anything else as a 1-particle jet
            jet_list += [[label] for label in unallocated_leaves]
        return jet_list


class CALEv3(FormJets.Agglomerative):
    _cheby_coeffs = CALEFunctions.cheby_coeff(lambda x: np.exp(-x), 50)
    default_params = {'Sigma': 0.1,
                      'InvPower': 1.,
                      'Cutoff': 0.02,
                      'WeightExponent': 1.,
                      'SeedGenerator': 'PtCenter',
                      'ToAffinity': 'exp',
                      'SeedIncrement': 1}
    permited_values = {'Sigma': Constants.numeric_classes['pdn'],
                       'InvPower': Constants.numeric_classes['pdn'],
                       'Cutoff': Constants.numeric_classes['rn'],
                       'WeightExponent': [0., Constants.numeric_classes['pdn']],
                       'SeedGenerator': ['PtCenter', 'Random', 'Unsafe'],
                       'ToAffinity': ['exp', 'inv'],
                       'SeedIncrement': Constants.numeric_classes['nn']}
    max_seeds = 15

    def _get_max_elements(self, n_inputs, n_unclustered):
        max_elements = n_inputs + int(0.5*(n_unclustered*(n_unclustered-1))) + 1
        max_elements += self.max_seeds
        return max_elements

    def _setup_clustering_functions(self):
        """
        Assigns internal functions needed in the clustering.
        Assumes the hyperparams have been set.
        """
        if self.SeedGenerator == 'PtCenter':
            self._seed_generator = CALEFunctions.pt_centers
        elif self.SeedGenerator == 'Random':
            self._seed_generator = CALEFunctions.random_centers
        elif self.SeedGenerator == 'Unsafe':
            self._seed_generator = CALEFunctions.unsafe_centers
        else:
            raise ValueError('SeedGenerator must be PtCenter, Unsafe or Random')
        self._calc_phys_distances2 = FormJets.ca_distances2
        if self.ToAffinity == 'exp':
            def affinity(distances2, sigma=self.Sigma):
                return np.exp(-(distances2)/(2*sigma**2))
        elif self.ToAffinity == 'inv':
            inv_multiplier = -self.InvPower*0.5
            def affinity(distances2, mult=inv_multiplier):
                return distances2**inv_multiplier
        else:
            raise ValueError("ToAffinity must be inv or exp")
        self._calc_affinity = affinity
        def laplacian(pts, distances2, affinities):
            return CALEFunctions.pt_laplacian(pts, distances2, affinities,
                weight_exponent=self.WeightExponent)
        self._calc_laplacian = laplacian

    def setup_internal(self):
        """ Runs before allocate """
        self._seed_labels = []
        self._n_seeds = self.SeedIncrement

    def _insert_new_seeds(self, n_seeds, mask):
        """
        Inserts new seeds into the clustering
        """
        seed_pxpypz, seed_ptrapphi = self._seed_generator(
                self.Energy[mask],
                self.Px[mask], self.Py[mask], self.Pz[mask],
                self.PT[mask], self.Rapidity[mask], self.Phi[mask],
                n_centers=n_seeds)
        current_max_label = np.max(self.Label)
        for i in range(len(seed_ptrapphi)):
            new_label = current_max_label + 1
            current_max_label += 1
            self._seed_labels.append(new_label)
            new_idx = self._next_free_row()
            self.Label[new_idx] = new_label
            # set all relationships to -1
            # and having Parent=-1
            self.Child1[new_idx] = -1
            self.Child2[new_idx] = -1
            self.Parent[new_idx] = -1
            self.Rank[new_idx] = -1
            # PT px py pz eta phi energy join_distance
            self.Energy[new_idx] = np.mean(self.Leaf_Energy)
            self.Px[new_idx] = seed_pxpypz[i, 0]
            self.Py[new_idx] = seed_pxpypz[i, 1]
            self.Pz[new_idx] = seed_pxpypz[i, 2]
            self.Size[new_idx] = 0.
            # it's easier conceptually to calculate pt, phi and rapidity
            # afresh than derive them
            # for some reason this must be unpacked then assigned
            self.Phi[new_idx] = seed_ptrapphi[i, 2]
            self.PT[new_idx] = seed_ptrapphi[i, 0]
            self.Rapidity[new_idx] = seed_ptrapphi[i, 1]

    def _remove_seeds(self):
        sorter = np.argsort(self.Label)
        seed_idxs = sorter[np.searchsorted(self.Label, self._seed_labels,
                                           sorter=sorter)]
        # no need to litrally delete the rows,
        # just wipe the label out
        self.Label[seed_idxs] = -1
        self._seed_labels.clear()
        self._available_idxs = [idx for idx in self._available_idxs if idx not in seed_idxs]
        self._available_mask[seed_idxs] = False

    def stopping_condition(self):
        # sometimes the seed leads to no jets
        if self._n_seeds > self.max_seeds:
            return True
        if len(self._available_idxs) < 2:
            return True
        if not np.any(self._unclustered_leaf_mask):
            return True
        return False

    def next_jets(self, fig=None, ax=None):
        """Sort the labels into exclusive jets"""
        # the assumption is that this function returns a list of
        # constituent labels for each jet
        # then another function is called to actually form the jets
        # As it forms the list it needs to keep track of the available
        # particles itself, but aside from adding and removing seeds,
        # there should eb no modification of the ints and floats
        jet_list = []
        # Cannot have more jets than input particles.
        # TODO this vs self._available_mask
        self._insert_new_seeds(self._n_seeds, self._unclustered_leaf_mask)
        #self._insert_new_seeds(self._n_seeds, self._available_mask)
        unallocated = set(self.Available_Label)
        mask = np.isin(self.Label, list(self.Available_Label) + self._seed_labels)
        # this will include the current seeds.
        laplacien = self._calc_laplacian(
                self.PT[mask], self.Rapidity[mask], self.Phi[mask])
        max_eigval = CALEFunctions.max_eigenvalue(laplacien)
        num_points = len(laplacien)
        found_content = False
        for seed_label in self._seed_labels:
            seed_location = self.Label[mask] == seed_label
            # TODO, should the range be from 0 to max_eigval?
            # do the eigenvalues actually go negative?
            wavelets = CALEFunctions.cheby_op(seed_location.astype(int),
                                              laplacien, [self._cheby_coeffs],
                                              (-max_eigval, max_eigval))[0]
            max_wavelet = np.max(wavelets[~seed_location])
            min_wavelet = np.min(wavelets[~seed_location])
            shifted_cutoff = (max_wavelet - min_wavelet)*self.Cutoff + min_wavelet
            below_cutoff = set(self.Label[mask][wavelets > shifted_cutoff])
            jet_labels = below_cutoff.intersection(unallocated)
            if len(jet_labels) > 1:  # if there is only one object, it's not a jet.
                found_content = True
                jet_list.append(list(jet_labels))
                unallocated -= jet_labels
            if len(unallocated) < 2:
                break  # one unallocated object
        # this batch of seeds is done.
        self._remove_seeds()
        if not found_content:
            self._n_seeds += self.SeedIncrement
        return jet_list

    def plot_next_jets(self, num_seeds=None, fig=None, ax=None):
        """Sort the labels into exclusive jets"""
        # the assumption is that this function returns a list of
        # constituent labels for each jet
        # then another function is called to actually form the jets
        # As it forms the list it needs to keep track of the available
        # particles itself, but aside from adding and removing seeds,
        # there should eb no modification of the ints and floats
        jet_list = []
        # plotting data ~~~~~~~~~~~
        made_jet = []
        seed_values = []
        stalk_plot_vars = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cannot have more jets than input particles.
        if num_seeds is not None:
            self._n_seeds = num_seeds
        self._insert_new_seeds(self._n_seeds, self._available_mask)
        unallocated = set(self.Available_Label)
        mask = np.isin(self.Label, list(self.Available_Label) + self._seed_labels)
        # this will include the current seeds.
        laplacien = self._calc_laplacian(
                self.PT[mask], self.Rapidity[mask], self.Phi[mask])
        max_eigval = CALEFunctions.max_eigenvalue(laplacien)
        num_points = len(laplacien)
        found_content = False
        for seed_label in self._seed_labels:
            seed_values.append(seed_label)  # plotting
            seed_location = self.Label[mask] == seed_label
            # TODO, should the range be from 0 to max_eigval?
            # do the eigenvalues actually go negative?
            wavelets = CALEFunctions.cheby_op(seed_location.astype(int),
                                              laplacien, [self._cheby_coeffs],
                                              (-max_eigval, max_eigval))[0]
            max_wavelet = np.max(wavelets[~seed_location])
            min_wavelet = np.min(wavelets[~seed_location])
            shifted_cutoff = (max_wavelet - min_wavelet)*self.Cutoff + min_wavelet
            below_cutoff = set(self.Label[mask][wavelets > shifted_cutoff])
            jet_labels = below_cutoff.intersection(unallocated)
            if len(jet_labels) > 1:  # if there is only one object, it's not a jet.
                made_jet.append(True)  # plotting
                found_content = True
                jet_list.append(list(jet_labels))
                unallocated -= jet_labels
            else:
                made_jet.append(False)  # plotting
            # Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            jet_content = np.isin(self.Label[mask], list(jet_labels))
            stalk_plot_vars.append((self.Rapidity[mask], self.Phi[mask], wavelets,
                                    jet_content, np.where(self.Label[mask] == seed_label)[0]))
            print(f"max_wavelet, {max_wavelet}, min_wavelet, {min_wavelet}, shifted_cutoff, {shifted_cutoff}")
            print(f"jet length = {sum(jet_content)}")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if len(unallocated) < 2:
                break  # one unallocated object
        # this batch of seeds is done.
        self._remove_seeds()
        self._n_seeds += self.SeedIncrement
        # plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt_clustering(self, seed_values, made_jet, jet_list, stalk_plot_vars)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not found_content:
            self._n_seeds += self.SeedIncrement
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


def plt_clustering(cluster_algo, seed_values, made_jet, jet_list, stalk_plot_vars):
    fig, ax1, ax2 = make_axes()
    colours = make_colours(len(seed_values))
    jet_colours = [c for c, j in zip(colours, made_jet) if j]
    plot_physics(ax1, cluster_algo.PT, cluster_algo.Rapidity, cluster_algo.Phi,
                 cluster_algo.Label, jet_list, seed_values, jet_colours)
    for i, args in enumerate(stalk_plot_vars):
        plot_points_on_stalks(ax2, colours[i], *args)


def make_axes():
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-np.pi, np.pi)
    ax1.set_title("Physics")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_title("Wavelets")
    return fig, ax1, ax2


def make_colours(n_jets):
    from matplotlib import cm
    return cm.rainbow(np.linspace(0, 1, n_jets))


def plot_physics(ax, pt, rapidity, phi, labels, jets, seed_labels, jet_colours):
    ax.scatter(rapidity, phi, s=5*np.sqrt(pt), c=['k'], alpha=.2)
    for colour, jet in zip(jet_colours, jets):
        mask = np.isin(labels, jet)
        for seed in seed_labels:
            mask[labels == seed] = False
        ax.scatter(rapidity[mask], phi[mask], s=5*np.sqrt(pt[mask]), color=[colour], alpha=.5)
    seed_mask = np.isin(labels, seed_labels)
    ax.scatter(rapidity[seed_mask], phi[seed_mask], s=10., c=['k'], marker='x')
    ax.set_xlabel('Rapidity')
    ax.set_ylabel('Phi')

current_z_max = 0
def plot_points_on_stalks(ax, colour, points_xs, points_ys, points_zs, idxs_selected, seed_point, norm_zs = False):
    if norm_zs:
        points_zs = 2*(points_zs - np.min(points_zs))/(np.max(points_zs) - np.min(points_zs)) - 1

    points_selected = np.zeros(len(points_xs), dtype = bool)
    points_selected[idxs_selected] = True

    ax.scatter(points_xs[points_selected], points_ys[points_selected], points_zs[points_selected], color=[colour], s = 2, marker = 'x')
    ax.scatter(points_xs[~points_selected], points_ys[~points_selected], points_zs[~points_selected], color=[colour], s = 2, marker = 'o')
    seed_mask = np.zeros(len(points_xs), dtype = bool)
    seed_mask[seed_point] = True
    ax.scatter(points_xs[seed_mask], points_ys[seed_mask], points_zs[seed_mask], c = ['k'], s = 10, marker = 'x')
    non_stalk_zs = points_zs[(~seed_mask)*points_selected]
    if len(non_stalk_zs) > 0:
        global current_z_max
        min_z = 0
        current_z_max = max(np.max(non_stalk_zs), current_z_max)
        ax.set_zlim(min_z, current_z_max)

    for i in range(len(points_xs)):
        ax.plot([points_xs[i], points_xs[i]], [points_ys[i], points_ys[i]], [0, points_zs[i]], color=colour, linewidth = 0.5)
    

FormJets.cluster_classes["CALE"] = CALE
FormJets.multiapply_input["CALE"] = CALE
FormJets.cluster_classes["CALEv2"] = CALEv2
FormJets.multiapply_input["CALEv2"] = CALEv2
FormJets.cluster_classes["CALEv3"] = CALEv3
FormJets.multiapply_input["CALEv3"] = CALEv3
FormJets.cluster_classes["CALECpp"] = CALECpp
FormJets.multiapply_input["CALECpp"] = CALECpp
