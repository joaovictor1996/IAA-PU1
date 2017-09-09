function [Wx,Wy,MSE]=trainMLP(E,Ce,S,Ta,alfa,X,D,epocaMax,MSETarget)

%E = Numero de entradas
%Ce = Numero de camadas escondidas
%S = Numero de saidas
%Ta = Taxa de aprendizado
%alfa = contante
 

[p1 N] = size(X);
bias = -1;

X = [bias*ones(1,N) ; X];

%Wx e Wy são pesos inicializados com valores aleatorios
%MSETemp armazena os valores dos erros calculados pela subtração de D por Y

Wx = rand(Ce,E+1);
WxAnt = zeros(Ce,E+1);
Tx = zeros(Ce,E+1);
Wy = rand(S,Ce+1);
Ty = zeros(S,Ce+1);
WyAnt = zeros(S,Ce+1);
DWy = zeros(S,Ce+1);
DWx = zeros(Ce,E+1);
MSETemp = zeros(1,epocaMax);


for i=1:epocaMax
    
k = randperm(N);
X = X(:,k);
D = D(:,k);

V = Wx*X;
Z = 1./(1+exp(-V));

S = [bias*ones(1,N);Z];
G = Wy*S;

Y = 1./(1+exp(-G));

E = D - Y;

mse = mean(mean(E.^2));
MSETemp(i) = mse;
disp(['epoch = ' num2str(i) ' mse = ' num2str(mse)]);
if (mse < MSETarget)
    MSE = MSETemp(1:i);
    return
end
 

df = Y.*(1-Y);
dGy = df .* E;

DWy = Ta/N * dGy*S';
Ty = Wy;
Wy = Wy + DWy + alfa*WyAnt;
WyAnt = Ty;

df= S.*(1-S);

dGx = df .* (Wy' * dGy);
dGx = dGx(2:end,:);
DWx = Ta/N* dGx*X';
Tx = Wx;
Wx = Wx + DWx + alfa*WxAnt;
WxAnt = Tx;
end

MSE = MSETemp;
end

