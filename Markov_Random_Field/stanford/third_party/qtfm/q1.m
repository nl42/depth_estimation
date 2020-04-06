function q = q1
% q1 is one of the three quaternion operators.
% q1 is usually denoted by i, but this symbol is used in Matlab to represent
% the complex operator (also represented by j).

% Copyright � 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

q = quaternion(1, 0, 0);

% Implementation note:  i cannot be overloaded because it is a built-in Matlab
% operator. I cannot be used because Matlab does not distinguish between upper
% and lower case.
