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
        if self.ToAffinity == 'exp':
            def laplacian(pts, rapidities, phis):
                return CALEFunctions.pt_laplacian(pts, rapidities, phis,
                    weight_exponent=self.WeightExponent, sigma=self.Sigma)
        elif self.ToAffinity == 'inv':
            def laplacian(pts, rapidities, phis):
                return CALEFunctions.pt_laplacian_inv(pts, rapidities, phis,
                    weight_exponent=self.WeightExponent, power=self.InvPower)
        else:
            raise ValueError('ToAffinity must be exp or inv')
        self._to_laplacian = laplacian

    def setup_internal(self):
        """ Runs before allocate """
        self._seed_labels = []

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
            # TODO optimise
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
            laplacien = self._to_laplacian(
                    self.PT[mask], self.Rapidity[mask], self.Phi[mask])
            max_eigval = CALEFunctions.max_eigenvalue(laplacien)
            num_points = len(laplacien)
            seed_jets = []
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
            laplacien = self._to_laplacian(
                    self.PT[mask], self.Rapidity[mask], self.Phi[mask])
            max_eigval = CALEFunctions.max_eigenvalue(laplacien)
            num_points = len(laplacien)
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

    for i in range(len(points_xs)):
        ax.plot([points_xs[i], points_xs[i]], [points_ys[i], points_ys[i]], [0, points_zs[i]], color=colour, linewidth = 0.5)
    

FormJets.cluster_classes["CALE"] = CALE
FormJets.multiapply_input["CALE"] = CALE
FormJets.cluster_classes["CALEv2"] = CALEv2
FormJets.multiapply_input["CALEv2"] = CALEv2
FormJets.cluster_classes["CALECpp"] = CALECpp
FormJets.multiapply_input["CALECpp"] = CALECpp
