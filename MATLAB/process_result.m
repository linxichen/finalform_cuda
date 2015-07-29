
V = csvread('../results/Vgrid.csv');
koptind = csvread('../results/koptind.csv');
kopt = csvread('../results/kopt.csv');
k_grid = csvread('../results/k_grid.csv')';
V = reshape(V,[300 25 25 7*7*2]);
koptind = reshape(koptind,[300 25 25  7*7*2]);
kopt = reshape(kopt,[300 25 25  7*7*2]);

plot(V(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1)))
plot(koptind(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1)))
plot(k_grid,kopt(:,randsample(25,1),randsample(25,1),randsample(7*7*2,1))-(1-0.025)*k_grid')