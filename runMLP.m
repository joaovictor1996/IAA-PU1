function Y=runMLP(X,Wx,Wy)

[p1 N] = size (X);
 
bias = -1;
 
X = [bias*ones(1,N) ; X];
 
V = Wx*X;
Z = 1./(1+exp(-V));
 
S = [bias*ones(1,N);Z];
G = Wy*S;
 
Y = 1./(1+exp(-G));
end

