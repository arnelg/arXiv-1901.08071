import numpy as np
# from math import factorial
import warnings

from scipy.special import binom
from scipy.linalg import eig
from numpy.core.multiarray import array

import qutip as qt

import channels


def factorial(x):
    if x >= 0.0:
        return np.math.factorial(x)
    else:
        return 0.0


def phasestate(s, r, phi0=0., fockdim=None):
    if fockdim is None:
        fockdim = s
    phim = phi0 + (2.0 * np.pi * r) / s
    n = np.arange(s)
    data = 1.0 / np.sqrt(s) * np.exp(1.0j * n * phim)
    data = np.hstack((data, np.zeros(fockdim-s)))
    return qt.Qobj(data)


def choi_to_kraus(q_oper, tol=1e-8):
    """
    Takes a Choi matrix and returns a list of Kraus operators.
    """
    vals, vecs = eig(q_oper.data.todense())
    vecs = [array(_) for _ in zip(*vecs)]
    shape = [np.prod(q_oper.dims[0][i]) for i in range(2)][::-1]
    kraus = [qt.Qobj(inpt=np.sqrt(val)*qt.vec2mat(vec, shape=shape),
             dims=q_oper.dims[0][::-1])
             for val, vec in zip(vals, vecs)]
    idx = np.where(vals > tol)[0]
    return np.array(kraus)[idx]


class CodeException(Exception):
    pass


class RotationalCode(object):
    """
    Generic code class used for inheritance.
    """

    def __init__(self, zero=None, one=None, plus=None, minus=None, N=None,
                 encoder=None, purity_threshold=1e-10):
        if encoder is not None:
            """
            def extract_ket(rho):
                vals, vecs = rho.eigenstates()
                idx = np.where(vals > purity_threshold)[0]
                if len(idx) > 1:
                    # raise CodeException("code not pure", vals[idx])
                    warnings.warn("code not pure " + str(vals[idx]))
                idx_max = np.argmax(vals)
                return(vecs[idx_max])
            zero = qt.ket2dm(qt.basis(2, 0))
            zero = extract_ket(qt.vector_to_operator(encoder
                               * qt.operator_to_vector(zero)))
            one = qt.ket2dm(qt.basis(2, 1))
            one = extract_ket(qt.vector_to_operator(encoder
                              * qt.operator_to_vector(one)))
            """
            zero = encoder*qt.basis(2, 0)
            one = encoder*qt.basis(2, 1)
        self._encoder = encoder
        if plus is None and zero is not None and one is not None:
            self._plus = (zero + one)/np.sqrt(2)
        elif plus is not None:
            self._plus = plus
        if minus is None and zero is not None and one is not None:
            self._minus = (zero - one)/np.sqrt(2)
        elif minus is not None:
            self._minus = minus
        if zero is None and plus is not None and minus is not None:
            self._zero = (plus + minus)/np.sqrt(2)
        elif zero is not None:
            self._zero = zero
        if one is None and plus is not None and minus is not None:
            self._one = (plus - minus)/np.sqrt(2)
        elif one is not None:
            self._one = one
        self._N = N
        self._name = 'rotcode'

    def encoder(self, kraus=False):
        if self._encoder is None:
            self._encoder = (self.zero*qt.basis(2, 0).dag()
                             + self.one*qt.basis(2, 1).dag())
        if kraus:
            return self._encoder
        else:
            return qt.sprepost(self._encoder, self._encoder.dag())

    def decoder(self, kraus=False):
        S = self.encoder(kraus=True)
        if kraus:
            return S.dag()
        else:
            return qt.sprepost(S.dag(), S)

    @property
    def name(self):
        return self._name

    @property
    def zero(self):
        return self._zero

    @property
    def one(self):
        return self._one

    @property
    def plus(self):
        return self._plus

    @property
    def minus(self):
        return self._minus

    @property
    def codewords(self):
        return self.zero, self.one

    @property
    def projector(self):
        # Projector onto code space P_code
        return self.zero*self.zero.dag() + self.one*self.one.dag()

    @property
    def logical_Z(self):
        # Q = self.identity-self.projector
        # return self.zero*self.zero.dag() - self.one*self.one.dag() + Q
        S = self.encoder(kraus=True)
        return S*qt.sigmaz()*S.dag()

    @property
    def logical_X(self):
        # Q = self.identity-self.projector
        # return self.zero*self.one.dag() + self.one*self.zero.dag() + Q
        S = self.encoder(kraus=True)
        return S*qt.sigmax()*S.dag()

    @property
    def logical_H(self):
        # Q = self.identity-self.projector
        # return (self.zero*self.zero.dag() + self.zero*self.one.dag()
        #         + self.one*self.zero.dag() - self.one*self.one.dag()
        #         )/np.sqrt(2) + Q
        S = self.encoder(kraus=True)
        return S*qt.hadamard_transform()*S.dag()

    @property
    def logical_Z_allspace(self):
        Q = self.identity-self.projector
        return self.logical_Z + Q

    @property
    def logical_X_allspace(self):
        Q = self.identity-self.projector
        return self.logical_X + Q

    @property
    def logical_H_allspace(self):
        Q = self.identity-self.projector
        return self.logical_H + Q

    @property
    def dim(self):
        # Hilbert space dimension
        return self.zero.dims[0][0]

    @property
    def identity(self):
        # Identity operator on full Hilbert space
        return qt.identity(self.dim)

    @property
    def annihilation_operator(self):
        return qt.destroy(self.dim)

    @property
    def number_operator(self):
        return qt.num(self.dim)

    def crot(self, ancilla):
        na = qt.tensor(self.number_operator, ancilla.identity)
        nb = qt.tensor(self.identity, ancilla.number_operator)
        return (1j*np.pi/(self.N*ancilla.N)*na*nb).expm()

    @property
    def N(self):
        return self._N

    def codeaverage(self, op):
        # Return average tr(P_code/2 op)
        return qt.expect(op, 0.5*self.projector)

    def codecheck(self, silent=False, atol=1e-6):
        # Check if code words are normalized
        x = np.abs(self.zero.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x-1)
        x = np.abs(self.one.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x-1)
        x = np.abs(self.plus.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x-1)
        x = np.abs(self.minus.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x-1)
        # Check if code words are orthogonal
        x = np.abs((self.zero.dag()*self.one).tr())
        if not np.isclose(x, 0, atol=atol):
            raise CodeException("code word not orthogonal", x)
        x = np.abs((self.plus.dag()*self.minus).tr())
        if not np.isclose(x, 0, atol=atol):
            raise CodeException("code word not orthogonal", x)
        if not silent:
            print("All good, buddy!")

    def commutator_check(self, silent=True, atol=1e-5):
        # Check if commutator is one
        a = self.annihilation_operator
        c = self.codeaverage(a*a.dag() - a.dag()*a)
        if not np.isclose(c, 1., atol=atol):
            raise CodeException("commutator not one", c)
        if not silent:
            print("All good, buddy!")

    def check_truncation(self, n):
        P = qt.Qobj(np.diag(np.hstack((np.zeros(n), np.ones(self.dim-n)))))
        return self.codeaverage(P)

    def check_rotationsymmetry(self, N=None, tol=1e-8):
        if N is None:
            N = self.N
        ck = self.zero.data.toarray()
        if not np.isclose(np.sum(np.abs(self.zero.data.toarray()[::2*N])**2),
                          1.0, rtol=tol, atol=tol):
            raise CodeException("zero not rotation symmetric")
        if not np.isclose(np.sum(np.abs(self.one.data.toarray()[N::2*N])**2),
                          1.0, rtol=tol, atol=tol):
            raise CodeException("one not rotation symmetric")

    def deleter(self, kraus=False):
        nothing = qt.basis(1, 0)
        k_list = [nothing*qt.basis(self.dim, i).dag() for i in range(self.dim)]
        if kraus:
            return k_list
        else:
            return qt.kraus_to_super(k_list)


class TrivialCode(RotationalCode):
    def __init__(self):
        zero = qt.basis(2, 0)
        one = qt.basis(2, 1)
        RotationalCode.__init__(self, zero=zero, one=one, N=1)
        self._name = 'trivialcode'


class PeggBarnett(RotationalCode):

    def __init__(self, N, s, fockdim, safety=True, novac=False):
        if safety and not s % 2*N == 0:
            raise ValueError('s needs to be a multiple of 2N.')
        zero = qt.Qobj()
        one = qt.Qobj()
        twoL = int(s/N)
        for k in range(novac, twoL):
            if k % 2 == 0:
                zero += qt.basis(fockdim, k*N)
            else:
                one += qt.basis(fockdim, k*N)
        # plus = (1 / np.sqrt(twoL)) * plus
        # minus = (1 / np.sqrt(twoL)) * minus
        zero = zero/zero.norm()
        one = one/one.norm()
        RotationalCode.__init__(self, zero=zero, one=one, N=N)
        self._name = 'peggbarnett'


class GCB(RotationalCode):

    def __init__(self, N, ck, fockdim, safety=True):
        if safety and len(ck) * N > fockdim:
            raise ValueError('N*len(ck) > fockdim')
        zero = qt.Qobj()
        one = qt.Qobj()
        for k in range(len(ck)):
            if k % 2 == 0:
                zero += ck[k]*qt.basis(fockdim, k*N)
            else:
                one += ck[k]*qt.basis(fockdim, k*N)
        zero = zero / zero.norm()
        one = one / one.norm()
        RotationalCode.__init__(self, zero=zero, one=one, N=N)
        self._name = 'gcb'


class BinCode(RotationalCode):

    def __init__(self, N, M, fockdim):
        if (M+2)*N > fockdim:
            raise ValueError('(M+2)*N > fockdim.')
        plus = qt.Qobj()
        minus = qt.Qobj()
        for k in range(M+2):
            bincoeff = np.sqrt(binom(M+1, k))
            plus += bincoeff*qt.basis(fockdim, k*N)
            minus += (-1)**k*bincoeff*qt.basis(fockdim, k*N)
        # plus = plus / plus.norm()
        # minus = minus / minus.norm()
        plus = (1 / np.sqrt(2**(M+1))) * plus
        minus = (1 / np.sqrt(2**(M+1))) * minus
        RotationalCode.__init__(self, plus=plus, minus=minus, N=N)
        self._name = 'binomial'


class PropellerCode(RotationalCode):

    def __init__(self, N, r, alpha, fockdim):
        zero = qt.Qobj()
        one = qt.Qobj()
        for m in range(2*N):
            phi = m*np.pi/N
            D = qt.displace(fockdim, alpha*np.exp(1j*phi))
            S = qt.squeeze(fockdim, r*np.exp(2j*(phi-np.pi/2)))
            blade = D*S*qt.basis(fockdim, 0)
            zero += blade
            one += (-1)**m*blade
        zero = zero/zero.norm()
        one = one/one.norm()
        self._alpha = alpha
        self._r = r
        RotationalCode.__init__(self, zero=zero, one=one, N=N)
        self._name = 'cat'


class GKPs(RotationalCode):

    def __init__(self, Delta, fockdim):
        zero = qt.Qobj()
        one = qt.Qobj()
        lim = int(2/Delta)
        for n1 in range(-lim, lim):
            for n2 in range(-2*lim, 2*lim):
                y = np.sqrt(np.pi/2)*n2
                Dy = qt.displace(fockdim, 1j*y)
                x = np.sqrt(np.pi/2)*(2*n1)
                Dx = qt.displace(fockdim, x)
                alpha = x + 1j*y
                fac = np.exp(-Delta**2*np.abs(alpha)**2)
                zero += fac*Dx*Dy*qt.basis(fockdim, 0)

                x = np.sqrt(np.pi/2)*(2*n1+1)
                Dx = qt.displace(fockdim, x)
                alpha = x + 1j*y
                fac = np.exp(-Delta**2*np.abs(alpha)**2)
                one += fac*Dx*Dy*qt.basis(fockdim, 0)
        zero = zero/zero.norm()
        one = one/one.norm()
        RotationalCode.__init__(self, zero=zero, one=one)
        self._name = 'gkp'


class MixedCode(RotationalCode):
    """
    A general code given by an encoding superoperator.
    Codewords can be basically anything, mixed, non-orthogonal, etc.
    """

    def __init__(self, encoder, N=None):
        self.encoder = encoder
        self.zero = qt.vector_to_operator(
                        encoder*qt.operator_to_vector(qt.basis(2, 0)))
        self.one = qt.vector_to_operator(
                        encoder*qt.operator_to_vector(qt.basis(2, 1)))
        self.plus = qt.vector_to_operator(encoder*qt.operator_to_vector(
                        (qt.basis(2, 0)+qt.basis(2, 1))/np.sqrt(2)))
        self.minus = qt.vector_to_operator(encoder*qt.operator_to_vector(
                        (qt.basis(2, 0)-qt.basis(2, 1))/np.sqrt(2)))
        RotationalCode.__init__(self, zero=zero, one=one, plus=plus,
                                minus=minus, N=N)

    @property
    def projector(self):
        # Projector onto code space P_code
        return self.zero + self.one

    @property
    def logical_Z(self):
        # Q = self.identity-self.projector
        return self.zero - self.one

    @property
    def logical_X(self):
        # Q = self.identity-self.projector
        return self.plus - self.minus

    @property
    def logical_H(self):
        # Q = self.identity-self.projector
        return (self.logical_Z+self.logical_X)/np.sqrt(2)


class SubsystemCode(RotationalCode):

    def __init__(self, code, ancilla_state):
        zero = qt.tensor(code.zero, ancilla_state)
        one = qt.tensor(code.one, ancilla_state)
        self._ancilla_state = ancilla_state
        RotationalCode.__init__(self, zero=zero, one=one, N=code.N)
