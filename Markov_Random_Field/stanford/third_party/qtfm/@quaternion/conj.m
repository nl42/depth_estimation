function c = conj(a, F)
% CONJ   Quaternion conjugate.
% (Quaternion overloading of Matlab standard function.)
%
% Implements three different conjugates:
%
% conj(X) or
% conj(X, 'hamilton') returns the quaternion conjugate.
% conj(X, 'complex')  returns the complex conjugate.
% conj(X, 'total')    returns the 'total' conjugate equivalent to
%                     conj(conj(X, 'complex'), 'hamilton')

% Copyright � 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 1
    F = 'hamilton'; % Supply the default parameter value.
end

if ~strcmp(F, 'hamilton') & ~strcmp(F, 'complex') & ~strcmp(F, 'total')
    error('Second parameter value must be ''hamilton'', ''complex'' or ''total''.')
end

switch F
    case 'hamilton'
        if ispure(a)
            c = -a;
        else
            c = quaternion(s(a), -x(a), -y(a), -z(a));
        end
    case 'complex'
        if ispure(a)
            c = quaternion(conj(x(a)), conj(y(a)), conj(z(a)));
        else
            c = conj(s(a)) + conj(v(a), 'complex');
        end
    case 'total'
        c = conj(conj(a, 'complex'));
    otherwise
        error('Bad value for second parameter.');
end
