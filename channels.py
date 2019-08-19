import numpy as np
import scipy.sparse

import qutip as qt

"""
Misc. useful functions
"""

vtensor = np.vectorize(qt.tensor)


"""
Classes
"""


class Channel():

    def __init__(self, kraus=None, channel_matrix=None, choi=None):
        self._kraus = kraus
        self._channel_matrix = channel_matrix
        self._choi = choi
        self._dim = None
        self._kraus_dual = None
        self._dual = None

    def __call__(self, other):
        return self.channel_matrix(other)

    @property
    def sysdim(self):
        if self._dim is None:
            self._dim = self.kraus[0].dims
        return self._dim

    @property
    def kraus(self):
        if self._kraus is not None:
            return self._kraus
        elif self._choi is not None:
            self._kraus = qt.choi_to_kraus(self._choi)
        elif self._channel_matrix is not None:
            self._choi = qt.super_to_choi(self._channel_matrix)
            self._kraus = qt.choi_to_kraus(self._choi)
        return self._kraus

    @property
    def channel_matrix(self):
        if self._channel_matrix is not None:
            return self._channel_matrix
        elif self._kraus is not None:
            self._channel_matrix = qt.kraus_to_super(self._kraus)
        elif self._choi is not None:
            self._channel_matrix = qt.choi_to_super(self._choi)
        return self._channel_matrix

    @property
    def choi(self):
        if self._choi is not None:
            return self._choi
        elif self._kraus is not None:
            self._choi = qt.kraus_to_choi(self._kraus)
        elif self._channel_matrix is not None:
            self._choi = qt.super_to_choi(self._channel_matrix)
        return self._choi

    @property
    def eigenvalues(self):
        return self._choi.eigenenergies()

    @property
    def tpcheck(self):
        tmp = self._choi.ptrace([0])
        ide = qt.identity(tmp.shape[0])
        print(qt.tracedist(tmp, ide))
        return tmp

    @property
    def istp(self):
        """ QuTiP istp check is too strcit. """
        tmp = self.choi.ptrace([0])
        ide = qt.identity(tmp.shape[0])
        return qt.isequal(tmp, ide, tol=1e-6)

    @property
    def iscp(self):
        """ QuTiP iscp check is too strict. """
        if not self.channel_matrix.iscp:
            lam = self.choi.eigenenergies()
            return (np.all(np.real(lam) > -1e-9)
                    and np.all(np.abs(np.imag(lam)) < 1e-9))
        else:
            return True

    @property
    def iscptp(self):
        return self.iscp and self.istp

    @property
    def dual(self):
        if self._dual is None:
            if self._kraus_dual is None:
                self._kraus_dual = [k.dag() for k in self._kraus]
            self._dual = Channel(kraus=self._kraus_dual)
        return self._dual


class IdentityChannel(Channel):

    def __init__(self, dim):
        k_list = [qt.identity(dim)]
        Channel.__init__(self, k_list)


class QubitAmplitudeDamping(Channel):

    def __init__(self, gamma):
        k0 = qt.Qobj([[1., 0.], [0., np.sqrt(1-gamma)]])
        k1 = qt.Qobj([[0., np.sqrt(gamma)], [0., 0.]])
        Channel.__init__(self, [k0, k1])


class LossChannel(Channel):
    """
    Sometimes errors arise if lmax is too big, probably due to overflow when
    taking the factorial??

    gamma = kappa tau
    """

    def __init__(self, gamma, dim, lmax):
        kt = gamma
        a = qt.destroy(dim)
        k_list = []
        x = np.exp(-kt)
        n_fac = n_pow_op(np.sqrt(x), 1, dim)
        for l in range(lmax):
            # c = np.sqrt(np.math.factorial(l))
            c = factorial_quotient(1-x, l)
            k_list.append(np.sqrt(c)*n_fac*a**l)
        Channel.__init__(self, kraus=k_list)
        self._kt = kt
        self._dim = dim

    def propagate(self, ancilla_dim, N, M):
        n = qt.num(ancilla_dim)
        return [(-1j*np.pi*k/(N*M)*n).expm() for k in range(len(self.kraus))]

    def propagate2(self, ancilla_dim, N, M):
        na = qt.num(self.sysdim)
        nb = qt.num(ancilla_dim)
        crot = (1j*np.pi/(N*M)*qt.tensor(na, nb)).expm()
        ida = qt.identity(ancilla_dim)
        return [qt.ptrace(qt.tensor(k.dag(), ida)*crot*qt.tensor(k, ida)*crot.dag(), 1)/(k.dag()*k).tr()
                for k in self.kraus]


class DephasingChannel(Channel):
    """
    Sometimes errors arise if lmax is too big, probably due to overflow when
    taking the factorial??

    gamma = kappa tau
    """

    def __init__(self, gamma, dim, lmax):
        def n_factor(base, powfac, polyfac, dim):
            op = np.zeros((dim, dim), dtype=np.complex_)
            for n in range(dim):
                op[n][n] = base**(n**powfac)*n**polyfac
            return qt.Qobj(op)
        kt = gamma
        n = qt.num(dim)
        k_list = []
        # n_fac = n_pow_op(np.exp(-0.5*kt), 2, dim)
        for l in range(lmax):
            c = np.sqrt(factorial_quotient(kt, l))
            # n_fac = n_factor(x, 2, l, dim)
            # k_list.append(c*n_fac*n**l)
            op = np.zeros((dim, dim), dtype=np.complex_)
            for n in range(dim):
                op[n][n] = c*np.exp(-0.5*kt*n**2)*n**l
            k_list.append(qt.Qobj(op))
        Channel.__init__(self, k_list)
        self._kt = kt


class POVM():

    def __init__(self, povm_elements, kraus=None, outcomes=None):
        self._outcomes = outcomes
        self._povm_elements = povm_elements
        self._dims = povm_elements[0].dims
        self._kraus = kraus

    @property
    def dim(self):
        return self._dims[0]

    @property
    def sysdim(self):
        return self.dim[0]

    @property
    def povm_elements(self):
        return self._povm_elements

    @property
    def kraus(self):
        if self._kraus is None:
            # compute kraus operators by doing Cholesky decomp of the POVM
            # elements. requires sksparse module
            from sksparse.cholmod import cholesky
            d = self._povm_elements[0].data.shape[0]
            reg = 1e-12*scipy.sparse.identity(d)
            self._kraus = [qt.Qobj(cholesky(a.data+reg).L()).dag()
                           for a in self.povm_elements]
        return self._kraus

    @property
    def outcomes(self):
        return self._outcomes

    @property
    def iscomplete(self):
        return qt.isequal(sum(self._povm_elements), qt.identity(self.dim),
                          tol=1e-12)


class PrettyGoodMeasurement(POVM):
    def __init__(self, code, noise):

        sigma = noise(code.projector)
        x = matrixf(sigma, lambda x: x**(-1/2), safe=True)
        povm_list = [
                     x*noise.channel_matrix(code.plus)*x,
                     x*noise.channel_matrix(code.minus)*x,
                    ]
        povm_list.append(code.identity - sum(povm_list))
        POVM.__init__(self, povm_list)


"""
Misc. functions
"""


def tensor_povm(povm1, povm2):
    povm_elements = [qt.tensor(a, b) for a in povm1.povm_elements
                     for b in povm2.povm_elements]
    return POVM(povm_elements)


def tensor_povm2(povm1, povm2):
    povm_elements = vtensor(np.array(povm1.povm_elements)[:, None],
                            np.array(povm2.povm_elements)).ravel()
    return POVM(povm_elements)


def n_pow_op(base, powfac, dim):
    """Creates an operator base**(n**powfac) where n is the number operator"""
    op = np.zeros((dim, dim), dtype=np.complex_)
    for n in range(dim):
        op[n][n] = base**(n**powfac)
    return qt.Qobj(op)


def factorial_quotient(numerator_base, l):
    # compute numertor_base^l/l!
    # useful when l is large and numertor_base < 1
    fac = 1
    for n in range(1, l+1):
        fac = fac*numerator_base/n
    return fac


def matrixf(rho, f, safe=False):
    # Apply f(rho) to diagonalizable matrix rho
    abstol = 1e-8
    vals, vecs = rho.eigenstates()
    out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
    for i, lam in enumerate(vals):
        if lam > abstol or not safe:
            out += f(lam)*vecs[i]*vecs[i].dag()
    return out
