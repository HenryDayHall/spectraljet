import numpy as np
from spectraljet import Components, FormJets
### minimal example of clusterng an event into jets

# create some data
mean_particles_per_jet = 10
n_jets = 5
particle_pxs = []
particle_pys = []
particle_pzs = []
particle_energies = []
# pt, rapidity and phi are optional arguments
# but if you have them supply them,
# it reduces run time
particle_pts = []
particle_rapidities = []
particle_phis = []

# for each jet, randomly generate some particles
for _ in range(n_jets):
    n_tracks = np.random.poisson(mean_particles_per_jet)
    pxs = np.random.normal(2*np.random.rand(), 2*np.random.rand(), n_tracks)
    particle_pxs += list(pxs)
    pys = np.random.normal(2*np.random.rand(), 2*np.random.rand(), n_tracks)
    particle_pys += list(pys)
    pzs = np.random.normal(3*np.random.rand(), 6*np.random.rand(), n_tracks)
    particle_pzs += list(pzs)
    energies = np.sqrt(pxs**2 + pys**2 + pzs**2)*(1+np.random.rand())
    particle_energies += list(energies)
    phis, pts = Components.pxpy_to_phipt(pxs, pys)
    particle_phis += list(phis)
    particle_pts += list(pts)
    rapidities = Components.ptpze_to_rapidity(pts, pzs, energies)
    particle_rapidities += list(rapidities)

# define the parameters of the clustering algorithm
# take a look at FormJets.Spectral.permitted_values
# to see all possible choices
spectral_clustering_parameters = {
                  'MaxMeanDist': 1.26,
                  'EigenvalueLimit': 0.4,
                  'Sigma': 0.15,
                  'CutoffKNN': 5,
                  'Beta': 1.4,
                  'ClipBeta': 1e-3,
                  'PhyDistance': 'angular',
                  'ExpofPhysDistance': 2.0,
                  'SingularitySuppression': 0.0001,
                  'Laplacian': 'symmetric_carried',
                  'EmbedDistance': 'root_angular',
                  'EmbedHardness': None,
                  'ExpofPTFormatInput': None,
                  'ExpofPTInput': 0,
                  'ExpofPTFormatAffinity': None,
                  'ExpofPTAffinity': 0.,
                  'ExpofPTFormatEmbedding': None,
                  'ExpofPTEmbedding': 0.,
                  }

# make a jet bundle, containing all clustering information
# for this event
jet_bundle = FormJets.Spectral.from_kinematics(particle_energies,
                                               particle_pxs,
                                               particle_pys,
                                               particle_pzs,
                                               particle_pts,  # optional
                                               particle_rapidities,  # optional
                                               particle_phis,  # optional
                                               dict_jet_params=spectral_clustering_parameters,
                                               run=True)  # tell it to cluster immediatly
seperate_jets = jet_bundle.split()  # break it into one object per jet
print(f"{len(seperate_jets)} jets have been formed")

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlabel("Rapidity")
ax.set_ylabel("Phi")
for jet in seperate_jets:
    ax.scatter(jet.Leaf_Rapidity, jet.Leaf_Phi, alpha=0.5, s=20*np.log(jet.Leaf_PT))
ax.set_title("Clustered jets")
plt.show()
