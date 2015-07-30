
V = csvread('../results/Vgrid.csv');
koptind = csvread('../results/koptind.csv');
kopt = csvread('../results/kopt.csv');
k_grid = csvread('../results/k_grid.csv')';
nk = length(k_grid);
V = reshape(V,[nk 25 25 7*7*2]);
koptind = reshape(koptind,[nk 25 25  7*7*2]);
kopt = reshape(kopt,[nk 25 25  7*7*2]);

figure
plot(V(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1)))
print -depsc2 V.eps

figure
plot(koptind(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1)))
print -depsc2 koptind.eps

figure
plot(k_grid,kopt(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1))-(1-0.025)*k_grid')
print -depsc2 kopt.eps
