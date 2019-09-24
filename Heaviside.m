%Heaviside function
function H=Heaviside(phi,alpha,nelx,nely,epsilon)
num_all=[1:(nelx+1)*(nely+1)]';
num1=find(phi>epsilon);
H(num1)=1;
num2=find(phi<-epsilon);
H(num2)=alpha;
num3=setdiff(num_all,[num1;num2]);
H(num3)=3*(1-alpha)/4*(phi(num3)/epsilon-phi(num3).^3/(3*(epsilon)^3))+(1+alpha)/2;
end