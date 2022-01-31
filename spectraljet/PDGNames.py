import numpy as np
import os

class Identities:
    """ """
    # see http://home.thep.lu.se/~torbjorn/pythia81html/ParticleDataScheme.html
    particle_attributes = {"id": int,
                           "name": str,      # - a character string with the name of the particle.
                                             # Particle and antiparticle names are stored
                                             # separately, with void returned when no antiparticle exists.
                           "antiName": str,  # - a character string with the name of the antiparticle.
                           "spinType": int,  # - the spin type, of the form 2 s + 1, with
                                             # special code 0 for entries of unknown or
                                             # indeterminate spin.
                           "chargeType": int,#  - three times the charge (to make it an integer).
                           "colType": int,   #  -the colour type, with 0 uncoloured, 1 triplet, 
                                             # 1 antitriplet and 2 octet.
                           "m0": float,      # - the nominal mass m_0 (in GeV).
                           "mWidth": float,  # - the width Gamma of the Breit-Wigner distribution (GeV).
                           "mMin": float,    # - the lower limit of the allowed mass range
                                             # generated by the Breit-Wigner (in GeV). Has no
                                             # meaning for particles without width, and would typically be 0 there.
                           "mMax": float,    # - the upper limit of the allowed mass range
                                             # generated by the Breit-Wigner (in GeV). If mMax < mMin
                                             # then no upper limit is imposed. Has no meaning for particles without
                                             # width, and would typically be 0 there.
                           "tau0": float,    # - the nominal proper lifetime tau_0 (in mm/c).
                            }
    def __init__(self, file_name=None):
        if file_name is None:
            dir_name = os.path.dirname(os.path.abspath(__file__))
            file_name = os.path.join(dir_name, "full_particle_data.txt")
        particle_data = []
        self.columns = {name: i for i, name in
                        enumerate(sorted(list(self.particle_attributes.keys())))}
        with open(file_name, 'r') as data_file:
            for line in data_file:
                # the lines contain lists of name="value"
                # seperted by space
                segments = line.split('"')
                values = [None for _ in self.columns]
                for seg_name, seg_value in zip(segments[::2], segments[1::2]):
                    # remove spaces and '=' sign from name
                    seg_name = seg_name.strip()[:-1] 
                    typ = self.particle_attributes[seg_name]
                    values[self.columns[seg_name]] = typ(seg_value)
                particle_data.append(values)
        self.particle_data = np.array(particle_data)
        #self.id_list = list(self.particle_data[:, self.columns['id']])

    def __getattr__(self, attr):
        # an s was added to the end to make plural
        if attr[:-1] in self.particle_attributes:
            required_cols = [self.columns["id"], self.columns[attr[:-1]]]
            return {pid: val for pid, val in self.particle_data[:, required_cols]}
        raise AttributeError

    @property
    def charges(self):
        """ """
        charge_types = self.chargeTypes
        part_charges = {i: charge_type/3 for i, charge_type in charge_types.items()}
        anti_charges = {-i: -charge_type/3 for i, charge_type in charge_types.items()}
        return {**part_charges, **anti_charges}

    @property
    def antiNames(self):
        """It antiparticle should be accesable from the negative pid too"""
        _antiNames = self.__getattr__('antiNames')
        neg_antiNames = {-key: name for key, name in _antiNames.items()}
        return {**_antiNames, **neg_antiNames}

    def __getitem__(self, idx):
        antiparticle = idx < 0
        pid = abs(idx)
        spinType = self.spinTypes[pid]
        chargeType = self.chargeTypes[pid]
        particle = {"id": idx,
                    "name": self.antiNames[pid] if antiparticle else self.names[pid],
                    "spin": None if spinType == 0 else (spinType - 1)/2,
                    "charge": (1-antiparticle*2)*chargeType/3,
                    "colType": self.colTypes[pid],
                    "mWidth": self.mWidths[pid],
                    "mMin": self.mMins[pid],
                    "mMax": self.mMaxs[pid],
                    "tau0": self.tau0s[pid]}
        return particle

    def __str__(self):
        return "(Particle attributes from Monte Carlo PID)"


class IDConverter(Identities):
    """ """
    def __getitem__(self, key):
        # override from Identities
        # if we don't find it in the dict return the key
        if key >= 0 :
            return self.names.get(key, key)
        else:
            return self.antiNames.get(key, key)


def match(pid_list, desired, partial=True):
    """
    

    Parameters
    ----------
    pid_list :
        param desired:
    partial :
        Default value = True)
    desired :
        

    Returns
    -------

    
    """
    desired = str(desired)
    converter = IDConverter()
    name_list = [str(converter[pid]) for pid in pid_list]
    if partial:
        return [desired in name for name in name_list]
    else:
        return [desired == name for name in name_list]

