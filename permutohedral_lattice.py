import numpy as np
from time import time
import logging

__author__ = 'Ido Freeman'
__email__ = "idofreeman@gmail.com"

log = logging.getLogger(__name__)

__all__ = ['PermutohedralLattice']


class HashTablePermutohedral(object):
    def __init__(self, kd_, vd_):
        """
        Constructor

        Attributes
        ----------
        kd_ : int
            the dimensionality of the position vectors on the hyperplane.
        vd_ : int
            the dimensionality of the value vectors
        """
        self.kd = kd_
        self.vd = vd_

        self.capacity = 2 ** 15
        self.filled = 0
        self.entries = [{'key_idx': -1, 'value_idx': -1} for _ in range(self.capacity)]
        self.keys = np.zeros((kd_ * self.capacity / 2), dtype='int16')
        self.values = np.zeros((vd_ * self.capacity / 2), dtype='float32')

    def size(self):
        return self.filled

    def get_keys(self):
        return self.keys

    def get_values(self):
        return self.values

    def lookup_offset(self, key, h, create=True):
        # Double hash table size if necessary
        if self.filled >= (self.capacity / 2) - 1:
            self._grow()

        # Find the entry with the given key
        while True:
            e = self.entries[h]
            if e['key_idx'] == -1:
                if not create:
                    # return not found
                    return -1
                # need to create an entry. Store the given key
                for i in range(self.kd):
                    self.keys[self.filled * self.kd + i] = key[i]
                e['key_idx'] = self.filled * self.kd
                e['value_idx'] = self.filled * self.vd
                self.entries[h] = e
                self.filled += 1
                return e['value_idx']

            # check if the cell has a matching key
            match = self.keys[e['key_idx']] == key[0]
            i = 1
            while i < self.kd and match:
                match = self.keys[e['key_idx'] + i] == key[i]
                i += 1

            if match:
                return e['value_idx']
            # increment the bucket with wraparound
            h += 1
            if h == self.capacity:
                h = 0

    def lookup(self, k, create=True):
        """
        Look up an object in the hash-table

        Attributes
        ----------
        k : numpy array
            A list of keys to match
        create : boolean, default True
            Whether a new entry should be created it nothing is matched

        Return
        ------
        The index of the first object in k, when k is completely matched
        """
        h = self._hash(k) % self.capacity
        offset = self.lookup_offset(k, h, create)
        if offset < 0:
            return None
        else:
            return offset

    def _hash(self, key):
        k = 0
        for i in range(self.kd):
            k += key[i]
            k *= 2531011
        return k

    def _grow(self):
        log.info('Resizing hash table')
        old_capacity = self.capacity
        self.capacity *= 2
        # Migrate the value vectors.
        new_values = np.zeros((self.vd * self.capacity / 2), dtype='float32')
        new_values[:self.vd * old_capacity / 2] = self.values
        self.values = new_values

        # Migrate the key vectors.
        new_keys = np.zeros((self.kd * self.capacity / 2), dtype='int16')
        new_keys[:self.kd * old_capacity / 2] = self.keys
        self.keys = new_keys

        # Migrate the table of indices.
        new_entries = [{'key_idx': -1, 'value_idx': -1} for _ in range(self.capacity)]
        for i in range(old_capacity):
            if self.entries[i]['key_idx'] == -1:
                continue
            h = self.hash(
                self.keys[self.entries[i]['key_idx']:self.entries[i]['key_idx'] + self.kd]
            ) % self.capacity
            while new_entries[h]['key_idx'] != -1:
                h += 1
                if h == self.capacity:
                    h = 0

            new_entries[h] = self.entries[i]
        self.entries = new_entries


class PermutohedralLattice(object):
    """
    Image filtering using a permutohedral lattice.
    Notice the method filter() does all the work
    """

    def __init__(self, d, vd, inp_len):
        """
        Initialise a new lattice object

        Attributes
        ----------
        d : int
            dimensionality of key vectors
        vd : int
            dimensionality of value vectors
        inp_len : int
            number of points in the input
        """
        self.d = d
        self.d1 = d + 1
        self.vd = vd
        self.inp_len = inp_len
        self.hash_table = HashTablePermutohedral(d, vd)

        self.elevated = np.zeros(self.d1, dtype='float32')
        self.scale_factor = np.zeros((d), dtype='float32')

        self.greedy = np.zeros(self.d1, dtype='int16')
        self.rank = np.zeros(self.d1, dtype='int16')
        self.barycentric = np.zeros((d + 2), dtype='float32')
        self.replay = [{'offset': 0, 'weight': 0.} for _ in range(inp_len * self.d1)]
        self.nReplay = 0
        self.canonical = np.zeros((self.d1 ** 2), dtype='int16')
        self.key = np.zeros(self.d1, dtype='int16')

        self.splat_scale = 1. / self.d1
        # compute the coordinates of the canonical simplex, in which
        # the difference between a contained point and the zero
        # remainder vertex is always in ascending order. (See pg.4 of paper.)
        for i in range(self.d1):
            for j in range(self.d1 - i):
                self.canonical[i * self.d1 + j] = i
            for j in range(self.d1 - i, self.d1):
                self.canonical[i * self.d1 + j] = i - self.d1

        expected_std = self.d1 * np.sqrt(2 / 3.)
        # Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
        for i in range(d):
            # the diagonal entries for normalization
            self.scale_factor[i] = expected_std / np.sqrt((i + 1) * (i + 2))
            """
                We presume that the user would like to do a Gaussian blur of standard deviation
                1 in each dimension (or a total variance of d, summed over dimensions.)
                Because the total variance of the blur performed by this algorithm is not d,
                we must scale the space to offset this.

                The total variance of the algorithm is (See pg.6 and 10 of paper):
                [variance of splatting] + [variance of blurring] + [variance of splatting]
                = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
                = 2d(d+1)(d+1)/3.

                So we need to scale the space by (d+1)sqrt(2/3).
            """
            # self.scale_factor[i] *= self.d1 * np.sqrt(2. / 3)

    @staticmethod
    def filter(inp, ref, debug=True):
        """
        Filter the image inp using the lattice defined by ref

        Attributes
        ----------
        inp : numpy array
            The image to filter, should have the shape (rows, cols, channels)
        ref : numpy array
            The lattice to use for filtering
        debug : boolean, default True
            Whether the function should print its current state and the different run times
        """
        run_times = [time()]

        # create lattice
        lattice = PermutohedralLattice(ref.shape[-1], inp.shape[-1] + 1, inp.shape[0] * inp.shape[1])
        run_times.append(time())

        if debug:
            log.info('Splatting...')
        col = np.zeros((inp.shape[-1] + 1), dtype='float32')
        col[-1] = 1.  # homogeneous coordinate

        # roll out ref to support running indexing
        ref_channels = ref.shape[-1]
        ref = ref.flatten()
        ref_ptr = 0
        for r in range(inp.shape[0]):  # height
            for c in range(inp.shape[1]):  # width
                col[:-1] = inp[r, c, :]
                lattice.splat(ref, ref_ptr, col)
                ref_ptr += ref_channels

        # Blur the lattice
        if debug:
            run_times.append(time())
            log.info('Blurring...')
        lattice.blur()

        # Slice from the lattice
        if debug:
            run_times.append(time())
            log.info('Slicing...')
        out = np.zeros_like(inp)

        lattice.begin_slice()
        for r in range(inp.shape[0]):  # height
            for c in range(inp.shape[1]):  # width
                lattice.slice(col)
                scale = 1. / col[-1]
                out[r, c, :] = col[:-1] * scale

        if debug:
            run_times.append(time())
            names = ['Init', 'Splat', 'Blur', 'Slice']
            log.info('Timing table (including prints)')
            for i in range(len(names)):
                log.info('{}: {}s'.format(names[i], run_times[i + 1] - run_times[i]))

        return out

    def splat(self, position, pos_idx, value):
        """
        Performs splatting with given position and value vectors

        Attributes
        ----------
        position : numpy array
            The lattice
        pos_idx : int
            The index of the relevant position on the grid
        """

        # first rotate position into the (d+1)-dimensional hyperplane
        self.elevated[self.d] = -self.d * position[pos_idx + self.d - 1] * self.scale_factor[self.d - 1]
        for i in range(self.d - 1, 0, -1):
            self.elevated[i] = self.elevated[i + 1] - \
                               i * position[pos_idx + i - 1] * self.scale_factor[i - 1] + \
                               (i + 2) * position[pos_idx + i] * self.scale_factor[i]
        self.elevated[0] = self.elevated[1] + 2 * position[pos_idx] * self.scale_factor[0]

        v = self.elevated * self.splat_scale
        v_ceil = np.ceil(v) * self.d1
        v_floor = np.floor(v) * self.d1
        self.greedy = np.where(v_ceil - self.elevated < self.elevated - v_floor, v_ceil, v_floor).astype('int16')

        sum = np.sum(self.greedy) / self.d1

        # reset rank
        self.rank *= 0
        # rank differential to find the permutation between this simplex and the canonical one.
        # (See pg. 3-4 in paper.)
        el_minus_gr = self.elevated - self.greedy
        for i in range(self.d):
            for j in range(i + 1, self.d1):
                if el_minus_gr[i] < el_minus_gr[j]:
                    self.rank[i] += 1
                else:
                    self.rank[j] += 1

        if sum > 0:
            # sum too large - the point is off the hyperplane.
            # need to bring down the ones with the smallest differential
            cond_mask = self.rank >= self.d1 - sum
            self.greedy[cond_mask] -= self.d1
            self.rank[cond_mask] -= self.d1
        elif sum < 0:
            # sum too small - the point is off the hyperplane
            # need to bring up the ones with largest differential
            cond_mask = self.rank < -sum
            self.greedy[cond_mask] += self.d1
            self.rank[cond_mask] += self.d1

        self.rank += sum

        # reset barycentric
        self.barycentric *= 0
        t = (self.elevated - self.greedy) * self.splat_scale
        # Compute barycentric coordinates (See pg.10 of paper.)
        for i in range(self.d1):
            self.barycentric[self.d - self.rank[i]] += t[i]
            self.barycentric[self.d1 - self.rank[i]] -= t[i]

        self.barycentric[0] += 1. + self.barycentric[self.d1]

        # Splat the value into each vertex of the simplex, with barycentric weights.
        for remainder in range(self.d1):
            # Compute the location of the lattice point explicitly
            # (all but the last coordinate - it's redundant because they sum to zero)
            self.key[:-1] = self.greedy[:-1] + self.canonical[remainder * self.d1 + self.rank[:-1]]

            # Retrieve pointer to the value at this vertex.
            hash_idx = self.hash_table.lookup(self.key, True)

            # Accumulate values with barycentric weight.
            # tmp = self.hash_table.values[hash_idx:hash_idx + self.vd] + self.barycentric[remainder] * value[
            #                                                                                           :self.vd]
            self.hash_table.values[hash_idx:hash_idx + self.vd] += self.barycentric[remainder] * value[:self.vd]

            # Record this interaction to use later when slicing
            self.replay[self.nReplay]['offset'] = hash_idx
            self.replay[self.nReplay]['weight'] = self.barycentric[remainder]
            self.nReplay += 1

    def slice(self, col):
        """
            Performs slicing out of position vectors. Note that the barycentric weights and the simplex
            containing each position vector were calculated and stored in the splatting step.
            We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)

        Attributes
        ----------
        col : numpy array
            A single position on the lattice
        """
        base = self.hash_table.get_values()
        col[:self.vd] = 0
        for i in range(self.d1):
            r = self.replay[self.nReplay]
            self.nReplay += 1
            for j in range(self.vd):
                col[j] += r['weight'] * base[r['offset'] + j]

    def blur(self, reverse=False):
        """
        Performs a Gaussian blur along each projected axis in the hyperplane
        
        Args
        ----
        reverse: bool, default False
           used for backprop (not supported in this version of the code) 
        """
        neighbour1 = np.zeros((self.d1), dtype='int16')
        neighbour2 = np.zeros((self.d1), dtype='int16')
        # new_vals = np.zeros((self.vd * self.hash_table.size()), dtype='float32')
        new_vals_idx = 0
        old_vals_idx = 0
        hash_table_base_idx = 0
        key = self.hash_table.get_keys()
        new_vals = np.zeros((self.vd * self.hash_table.size()), dtype='float64')
        old_vals = np.copy(self.hash_table.values)

        # For each of d+1 axes, reverse takes care of the gradient computation during the backward pass
        r = range(self.d, -1, -1) if reverse else range(self.d1)
        for j in r:
            # log.info(j)
            # For each vertex in the lattice
            for i in range(self.hash_table.size()):  # blur point i in dimension j
                neighbour1[:self.d] = key[i * self.d:(i + 1) * self.d] + 1
                neighbour2[:self.d] = key[i * self.d:(i + 1) * self.d] - 1

                neighbour1[j] = key[j + i * self.d] - self.d
                neighbour2[j] = key[j + i * self.d] + self.d  # keys to the neighbors along the given axis.

                old_val_offset = old_vals_idx + i * self.vd
                new_val_offset = new_vals_idx + i * self.vd

                vm1_idx = self.hash_table.lookup(neighbour1, False)  # look up first neighbor
                if vm1_idx is not None:
                    vm1_idx -= hash_table_base_idx + old_vals_idx
                else:
                    vm1_idx = None

                vp1_idx = self.hash_table.lookup(neighbour2, False)  # look up second neighbor
                if vp1_idx is not None:
                    vp1_idx -= hash_table_base_idx + old_vals_idx
                else:
                    vp1_idx = None

                # Mix values of the three vertices
                if vm1_idx is None:
                    vm1_val = 0
                else:
                    vm1_val = old_vals[vm1_idx:vm1_idx + self.vd]

                if vp1_idx is None:
                    vp1_val = 0
                else:
                    vp1_val = old_vals[vp1_idx:vp1_idx + self.vd]

                # applies the convolution with a 1d kernel [1, 2, 1]
                # self.hash_table.values[new_val_offset:new_val_offset + self.vd] = \
                new_vals[new_val_offset:new_val_offset + self.vd] = \
                    .25 * (vm1_val + vp1_val) + \
                    .5 * old_vals[old_val_offset:old_val_offset + self.vd]

            tmp = new_vals_idx
            new_vals_idx = old_vals_idx
            old_vals_idx = tmp

            tmp = new_vals
            new_vals = old_vals
            old_vals = tmp
            # the freshest data is now in oldValue, and newValue is ready to be written over

        self.hash_table.values = old_vals

    def begin_slice(self):
        """
        Prepare for slicing
        """
        self.nReplay = 0
