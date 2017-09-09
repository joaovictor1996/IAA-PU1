%E = Numero de entradas
%Ce = Numero de camadas escondidas
%S = Numero de saidas
%Ta = Taxa de aprendizado
%epoca = Numero de iterações
%MSEmin = Erro Minimo

E = 3;
Ce = 10;
S = 2;

Ta = 5;
alfa = 0.00000000001;

epoca = 25000;
MSEmin = 1e-10;

%X = Matriz com os valores de entrada
%D = Matriz com os valores de saida desejados

X = [3.0 3.0 3.0 3.0 2.0 0.5 0.5 0.5 0.3 0.2 0.2 0.2 3.0 0.2 3.0 0.7 0.7 0.4 3.0 1.0; %frente
     3.0 2.0 1.0 0.5 3.0 3.0 0.5 3.0 3.0 2.0 2.0 3.0 3.0 0.2 2.0 0.5 2.2 0.2 0.3 0.4; %direita 
     3.0 2.0 1.0 0.5 3.0 3.0 3.0 0.5 0.5 1.0 2.0 2.0 0.2 2.0 1.0 0.3 3.0 3.0 3.0 3.0];   %esquerda
D = [0.4 0.4 0.4 0.4 0.4 -0.1 0.1 -0.2 -0.2 0.1 0.4 0.1 0.1 0.3 0.5 0.1 0.5 0.4 0.5 0.5; %direita
     0.4 0.4 0.4 0.4 0.4 0.2 -0.2 0.1 0.1 0.3 0.1 0.4 0.3 -0.1 0.5 0.4 -0.2 0.1 0.3 0.3]; %esquerda

%Chamando a função de treinamento
 
[Wx,Wy,MSE]=trainMLP(E,Ce,S,Ta,alfa,X,D,epoca,MSEmin);

save('weights','Wx','Wy');
semilogy(MSE);

%Chamando a função runMPL

Y = runMLP(X,Wx,Wy);