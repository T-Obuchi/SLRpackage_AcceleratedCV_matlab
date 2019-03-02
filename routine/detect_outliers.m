function [MASK,cMASK]=detect_outliers2(dataV,errV)

Ldata=length(dataV);

datadiff1=zeros(Ldata,1);
datadiff2=zeros(Ldata,1);
datadiff3=zeros(Ldata,1);

datadiff1(2:end  ) = ( dataV(2:end)-dataV(1:end-1) )./(2*errV(1:end-1));
datadiff2(3:end  ) = ( dataV(3:end)-dataV(1:end-2) )./(3*errV(1:end-2));
datadiff3(4:end  ) = ( dataV(4:end)-dataV(1:end-3) )./(4*errV(1:end-3));

MASK1 = abs(datadiff1(:))<1;
MASK2 = abs(datadiff2(:))<1;
MASK3 = abs(datadiff3(:))<1;
MASK  = logical(MASK1.*MASK2.*MASK3);
cMASK = not(MASK);

end