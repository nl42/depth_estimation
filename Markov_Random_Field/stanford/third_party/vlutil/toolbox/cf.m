function h=cf(h0) ;
% CF  Copy figure
%   H=CF() creates a duplicate of the current figure.
%
%   H=CF(H0) creates a duplicate of the figure whose handle is H0.

if nargin < 1
  h0 = gcf ;
end

h=zeros(size(h0)) ;
for i=1:numel(h0) 
  h(i) = copyobj(h0(i),get(h0(i),'Parent')) ;
end

if nargout < 1
  clear h ;
end