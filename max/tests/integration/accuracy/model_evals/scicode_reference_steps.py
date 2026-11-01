# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Reference code for the three SciCode steps the official harness never generates.

Steps 13.6, 62.1 and 76.3 have no test cases. The official harness
(scicode-bench/SciCode ``eval/scripts/gencode.py``) skips generating them and
splices these reference implementations from its ``eval/data/`` directory into
the solution so later steps of the same problem can call them. That is why the
benchmark scores 338 of the dataset's 341 sub-steps.

Copied verbatim from https://github.com/scicode-bench/SciCode at commit
e3158ea011d4235245a547460d3688d7ccbf9900 (Apache License 2.0).
"""

REFERENCE_STEP_CODE: dict[str, str] = {
    "13.6": '''# code 1.6
class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out):
        """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y component of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.

        """

        self.n_grid = n_grid
        self.n_vars = 7
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta

        self.x      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[:,None,None]
        self.y      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,:,None]
        self.z      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,None,:]
        self.r      = np.sqrt(self.x**2+self.y**2+self.z**2)


        # set up all variables common to both approaches
        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))


        self.t = 0.0
''',
    "62.1": """class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\\n{matrix}\\n")
            else:
                print(f"{key}:\\n{matrix.toarray()}\\n")

class EnlargedBlock:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\\n{matrix}\\n")
            else:
                print(f"{key}:\\n{matrix.toarray()}\\n")
""",
    "76.3": """def generate_dna(N: int, PWM: dict) -> tuple:
    '''
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    '''
    p = random.randint(0, N-1)

    nucleotide = "ACGT"
    uni_weights = [0.25,0.25,0.25,0.25] #uniform distribution
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))

    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join(random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0]
                         for i in range(len(PWM['A'])))

    complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join(complement[nuc] for nuc in reversed_seq if nuc in complement)

    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N-p] + spiked_seq + reverse_complement[N-p:]

    return p, new_seq, new_seq_rc
""",
}
