function finalRad= Projeto1(obj)
 
load weights.mat;
 
 
 
while(1)
     
     
    %[x y th]= OverheadLocalizationCreate(obj)
     
    dd = ReadSonar(obj,1)
    if (isempty(dd))
        dd = 3;
    end
        
         
    df = ReadSonar(obj,2)
     if (isempty(df))
        df = 3;
    end
    de = ReadSonar(obj,3)
     
    if (isempty(de))
        de = 3;
    end
    %dt = ReadSonar(obj,4)
     
    X = [df;dd;de]
     
    Y = runMLP(X,Wx,Wy);
     
    SetDriveWheelsCreate(obj,Y(1),Y(2));
     
    [x y th]= OverheadLocalizationCreate(obj);
    plot(x,y,'r*');
     
    pause(1);
%endfunction [ output_args ] = Untitled2( input_args )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


end

