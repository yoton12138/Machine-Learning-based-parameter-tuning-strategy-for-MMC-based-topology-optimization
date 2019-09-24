%Forming void domain
function [tmpvoid]=tvoid(LSgridx,LSgridy)
a=abs(2-LSgridx)-1.2;
b=abs(2-LSgridy)-1.2;
tmpvoid=max(a,b);
end