function p = vector(q)
% VECTOR   Quaternion vector part. Synonym of V.

% Copyright � 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

p = v(q);
