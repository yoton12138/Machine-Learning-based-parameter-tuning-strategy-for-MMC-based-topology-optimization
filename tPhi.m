%Forming Phi_i for each component
function [tmpPhi]=tPhi(xy,LSgridx,LSgridy,p)
st=xy(7);
ct=sqrt(abs(1-st*st));
x1=ct*(LSgridx - xy(1))+st*(LSgridy - xy(2));
y1=-st*(LSgridx - xy(1))+ct*(LSgridy - xy(2));
bb=(xy(5)+xy(4)-2*xy(6))/2/xy(3)^2*x1.^2+(xy(5)-xy(4))/2*x1/xy(3)+xy(6);
tmpPhi= -((x1).^p/xy(3)^p+(y1).^p./bb.^p-1);
end