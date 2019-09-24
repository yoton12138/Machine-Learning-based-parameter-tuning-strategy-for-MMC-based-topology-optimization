%Element stiffness matrix
function [KE] = BasicKe(E,nu, a, b,h)
k = [-1/6/a/b*(nu*a^2-2*b^2-a^2),  1/8*nu+1/8, -1/12/a/b*(nu*a^2+4*b^2-a^2), 3/8*nu-1/8, ...
    1/12/a/b*(nu*a^2-2*b^2-a^2),-1/8*nu-1/8,  1/6/a/b*(nu*a^2+b^2-a^2),   -3/8*nu+1/8];
KE = E*h/(1-nu^2)*...
    [ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
    k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
    k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
    k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
    k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
    k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
    k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
    k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
end