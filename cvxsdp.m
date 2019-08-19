function R=cvxsdp(C, physdim, codedim)

physdim = double(physdim);
codedim = double(codedim);
dim = codedim*physdim;

C = reshape(C, dim, dim);

cvx_begin sdp quiet
    variable X(dim, dim) hermitian
    maximise real(trace(X*C))
    subject to
        X >= 0;
        TrX(X, [2], [physdim, codedim]) == eye(physdim);
cvx_end

R = full(X);
