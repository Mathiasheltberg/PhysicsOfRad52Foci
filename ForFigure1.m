clear all; close all; clc

ts = 0.02; MsTr = []; count = 0; LT = [];
for i0 = 1:8
    A = importdata(['Cell',num2str(i0),'.trxyt']);
    A1 = unique(A(:,1)); %% Define all unique traces
    [dd dd2] = unique(A(:,1));
    for u1 = 1:length(dd)
        ins = find(A(:,1)==dd(u1));
        if (length(ins)>1)
            count = count + 1;
            cx = A(ins,2); cy = A(ins,3);
            for ms = 1:length(ins)
                MsTr{ms}{count} = sqrt( (cx(1+ms:end) - cx(1:end-ms)).^2 + (cy(1+ms:end) - cy(1:end-ms)).^2 );
            end
            LT = [LT; length(cx)-1];
        end
    end
end

% fun = @(x) exp(-x.^2/(4*D*ts));
% q = 1-integral(fun,-0.15,0.15);
% plot(hx,q*exp(-q*hx));

h = histogram(LT);
h.BinEdges = [0.5:1:max(LT)]; h.FaceColor = [0.0 0.7 0.0]; h.EdgeColor = [0.1 0.9 0.1]; h.FaceAlpha = 0.05; h.EdgeAlpha = 0.05;
h.Normalization = 'count'; h.LineWidth = 1; hold on; goodplot
hy = h.Values; hx = h.BinEdges(1:end-1)+0.5*h.BinWidth; hw = h.BinWidth;
figure; semilogy(hx,hy,'-*','LineWidth',3); goodplot
figure
D_LL = []; D_CS = []; gg = hot(8); L = [];
for i = 1:6
    DR = [];
    for j = 1:length(MsTr{i})
        DR = [DR; MsTr{i}{j}];
    end
    L = [L; length(DR)];
   
    h = histogram(DR);
    h.BinEdges = [0:0.02:max(DR)]; h.FaceColor = [0.0 0.7 0.0]; h.EdgeColor = [0.1 0.9 0.1]; h.FaceAlpha = 0.05; h.EdgeAlpha = 0.05;
    h.Normalization = 'PDF'; h.LineWidth = 1; hold on; goodplot
    hy = h.Values; hx = h.BinEdges(1:end-1)+0.5*h.BinWidth; hw = h.BinWidth;
    
    A10l = 0.6-0.05*i; D10l = 0.3-0.02*i; D20l = 1.2-0.1*i;
    A10c = rand; D10c = 0.05; D20c = rand;
    for te = 1:10000
        D1l = D10l + 0.01*randn;  D2l = D20l + 0.01*randn; A1l = A10l + 0.01*randn;
        if (A1l > 1 || A1l < 0); A1l = rand; end
        if (D2l < 0); D2l = rand; end
        if (D1l < 0); D1l = rand; end
        
        P1 = A10l*DR.*exp(-DR.^2/(4*D10l*ts*i))/(2*D10l*ts*i) + (1-A10l)*DR.*exp(-DR.^2/(4*D20l*ts*i))/(2*D20l*ts*i);
        P2 = A1l*DR.*exp(-DR.^2/(4*D1l*ts*i))/(2*D1l*ts*i) + (1-A1l)*DR.*exp(-DR.^2/(4*D2l*ts*i))/(2*D2l*ts*i);
        P = prod(P2./P1);
        if (P > 1); D10l = D1l; D20l = D2l; A10l = A1l; end
        
        D1c = D10c + 0.01*randn;  D2c = D20c + 0.01*randn; A1c = A10c + 0.01*randn;
        E1 = A10c*hx.*exp(-hx.^2/(4*D10c*ts*i))/(2*D10c*ts*i) + (1-A10c)*hx.*exp(-hx.^2/(4*D20c*ts*i))/(2*D20c*ts*i);
        E2 = A1c*hx.*exp(-hx.^2/(4*D1c*ts*i))/(2*D1c*ts*i) + (1-A1c)*hx.*exp(-hx.^2/(4*D2c*ts*i))/(2*D2c*ts*i);
        ChS1 = sum( ((hy-E1).^2)./E1);
        ChS2 = sum( ((hy-E2).^2)./E2);
        if (ChS2 < ChS1)
            D20c = D2c;
            D10c = D1c;
            A10c = A1c;
        end
    end
    
    if (i==1)
        E1 = A10l*hx.*exp(-hx.^2/(4*D10l*ts*i))/(2*D10l*ts*i) + (1-A10l)*hx.*exp(-hx.^2/(4*D20l*ts*i))/(2*D20l*ts*i);
        E1 = E1*length(hy)*hw
        ChS1 = sum( ((hy-E1).^2)./E1)
        chi2cdf(ChS1,length(hy)-4)
        Rsq = 1-sum((hy-E1).^2)/sum((hy-mean(hy)).^2)
        %          h = histogram(DR);
        %             h.BinEdges = [0:0.02:0.99]; h.FaceColor = [0.0 0.7 0.0]; h.EdgeColor = [0.1 0.9 0.1]; h.FaceAlpha = 0.1; h.EdgeAlpha = 0.1;
        %             Normalization = 'count'; h.LineWidth = 1; hold on; goodplot
        %             hy1 = h.Values; hx1 = h.BinEdges(1:end-1)+0.5*h.BinWidth;
        %          co = hy1(1)./hy(1);
        %          ChS1 = ChS1*co
        DB = sum(DR.^2)/(4*ts*length(DR));
        E1 = hx.*exp(-hx.^2/(4*DB*ts*i))/(2*DB*ts*i);
        E1 = E1*length(hy)*hw;
        ChS1 = sum( ((hy-E1).^2)./E1)
        chi2cdf(ChS1,length(hy)-2)
        Rsq = 1-sum((hy-E1).^2)/sum((hy-mean(hy)).^2)
        %          ChS1 = ChS1*co
        %          plot(hx,co*hx.*(A10l/(2*D10l*ts*i).*exp(-hx.^2./(4*D10l*ts*i)) + (1-A10l)/(2*D20l*ts*i).*exp(-hx.^2./(4*D20l*ts*i))),'color',gg(i,:),'LineWidth',3);
        
    end
    
    if (D10c >D20c); D20 = D20c; D20c = D10c; D10c = D20; A10c = 1-A10c; end
    if (D10l >D20l); D20 = D20l; D20l = D10l; D10l = D20; A10l = 1-A10l; end
    
    D_CS = [D_CS; A10c D10c 1-A10c D20c];
    D_LL = [D_LL; A10l D10l 1-A10l D20l];
    plot(hx,hy,'-*','color',gg(i,:));
    hx = linspace(0,1.5,1000);
    plot(hx,hx.*(A10l/(2*D10l*ts*i).*exp(-hx.^2./(4*D10l*ts*i)) + (1-A10l)/(2*D20l*ts*i).*exp(-hx.^2./(4*D20l*ts*i))),'color',gg(i,:),'LineWidth',3);
    
    
    
end

nn = 6;
x = linspace(1,nn,nn)'*ts;
figure;
semilogy(x,D_LL(1:nn,2),'-*c','LineWidth',3); hold on;
%semilogy(x,D_CS(1:4,2),'-*b','LineWidth',3);goodplot
semilogy(x,D_LL(1:nn,4),'-om','LineWidth',3); hold on;
%semilogy(x,D_CS(1:4,4),'-or','LineWidth',3);goodplot
goodplot
axis([0 0.14 0 1.3])
[a b] = polyfit(x,D_LL(1:nn,4),1);
xp = [0;x];
plot(xp,a(2)+a(1)*xp,'--r','LineWidth',3);
D1 = (-a(1)/4+a(2))/2;
a(2)
D1
[a b] = polyfit(x,D_LL(1:nn,2),1);
plot(xp,a(2)+a(1)*xp,'--b','LineWidth',3);
D2 = (-a(1)/4+a(2))/2;
a(2)
legend('L estimate (slow)','L estimate (fast)','Expected decay D1','Expected decay D2')

figure;
plot(D_LL(1:6,1),'-*','LineWidth',3); hold on;
plot(D_LL(1:6,3),'-*','LineWidth',3); goodplot
legend('Fraction fast','Fraction slow')
axis([0 7 0 2])
x = linspace(0,6,6)';
figure
plot(D_LL(1:6,1)./D_LL(1:6,3),'-*m','LineWidth',3); hold on;

plot(x,1.4*exp(-x*0.4*1.1*(exp(-D2^0.6)-exp(-D1^0.6))),'--*k','LineWidth',2); goodplot
axis([0 7 0 2])
legend('Fraction slow/Fraction fast','Expected decay')