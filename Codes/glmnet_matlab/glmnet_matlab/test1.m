clear ; close all ; clc ;

% Cox:
   N=1000; p=30;
   nzc=p/3;
   x=randn(N,p);
   beta=randn(nzc,1);
   fx=x(:,1:nzc)*beta/3;
   hx=exp(fx);
   ty=exprnd(1./hx,N,1);
   tcens=binornd(1,0.3,N,1);
   y=cat(2,ty,1-tcens);
   fit=glmnet(x,y,'cox');
   glmnetPlot(fit);