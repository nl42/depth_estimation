function a = fix(q)
% FIX Round towards zero.
% (Quaternion overloading of standard Matlab function.)

% Copyright � 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

if ispure(q)
    a = quaternion(           fix(x(q)), fix(y(q)), fix(z(q)));
else
    a = quaternion(fix(s(q)), fix(x(q)), fix(y(q)), fix(z(q)));
end
