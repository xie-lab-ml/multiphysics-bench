function out = Elder_v1(parm)

parm_NN=parm.NN;

parm_Sc_A=parm.Sc_A;
parm_Sc_x0=parm.Sc_x0;
parm_Sc_y0=parm.Sc_y0;
parm_Sc_sigma=parm.Sc_sigma;

%
% Elder_v1.m
%
% Model exported on Apr 2 2025, 22:35 by COMSOL 6.2.0.290.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('E:\DATA\Elder');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('dl', 'PorousMediaFlowDarcy', 'geom1');
model.component('comp1').physics.create('tds', 'DilutedSpeciesInPorousMedia', {'c'});

model.study.create('std1');
model.study('std1').create('time', 'Transient');
model.study('std1').feature('time').setSolveFor('/physics/dl', true);
model.study('std1').feature('time').setSolveFor('/physics/tds', true);

model.param.set('L', '150[m]');
model.param.descr('L', 'Basin depth');
model.param.set('rho0', '1000[kg/m^3]');
model.param.descr('rho0', 'Pristine water density');
model.param.set('rho_s', '1200[kg/m^3]');
model.param.descr('rho_s', 'Brine density');
model.param.set('c0', '0[mol/m^3]');
model.param.descr('c0', 'Zero salt concentration');
model.param.set('c_s', '1[mol/m^3]');
model.param.descr('c_s', 'Normalized salt concentration');
model.param.set('beta', '(rho_s-rho0)/(c_s-c0)');
model.param.descr('beta', 'Increase in density due to salt concentration');
model.param.set('p0', '0[atm]');
model.param.descr('p0', 'Reference pressure');
model.param.set('mu', '1e-3[Pa*s]');
model.param.descr('mu', 'Dynamic viscosity');
model.param.set('kappa', '500[mD]');
model.param.descr('kappa', 'Permeability');
model.param.set('epsilon', '0.1');
model.param.descr('epsilon', 'Porosity');
model.param.set('D_L', '3.56e-6[m^2/s]');
model.param.descr('D_L', 'Molecular diffusion');
model.param.set('Pe', 'beta*(c_s-c0)*g_const*kappa*L/(mu*epsilon*D_L)');
model.param.descr('Pe', 'Peclet number');
model.param.set('sigma', parm_Sc_sigma);
model.param.descr('sigma', '');
model.param.set('x0', parm_Sc_x0);
model.param.descr('x0', '');
model.param.set('y0', parm_Sc_y0);
model.param.descr('y0', '');
model.param.set('A', parm_Sc_A);
model.param.descr('A', '');

model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size', {'2*L' 'L'});
model.component('comp1').geom('geom1').feature('r1').set('base', 'center');
model.component('comp1').geom('geom1').run('r1');
model.component('comp1').geom('geom1').create('pt1', 'Point');
model.component('comp1').geom('geom1').feature('pt1').setIndex('p', 'L/2', 1);
model.component('comp1').geom('geom1').run('fin');

model.component('comp1').physics('dl').prop('GravityEffects').set('IncludeGravity', true);
model.component('comp1').physics('dl').feature('porous1').feature('fluid1').set('rho_mat', 'userdef');
model.component('comp1').physics('dl').feature('porous1').feature('fluid1').set('mu_mat', 'userdef');
model.component('comp1').physics('dl').feature('porous1').feature('fluid1').set('rho', 'rho');
model.component('comp1').physics('dl').feature('porous1').feature('fluid1').set('mu', 'mu');

model.component('comp1').variable.create('var1');
model.component('comp1').variable('var1').set('rho', 'rho0+beta*c*(c>0)');
model.component('comp1').variable('var1').descr('rho', [native2unicode(hex2dec({'6c' '34'}), 'unicode')  native2unicode(hex2dec({'5b' 'c6'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ]);

model.component('comp1').physics('dl').feature('porous1').feature('pm1').set('epsilon_mat', 'userdef');
model.component('comp1').physics('dl').feature('porous1').feature('pm1').set('kappa_mat', 'userdef');
model.component('comp1').physics('dl').feature('porous1').feature('pm1').set('epsilon', 'epsilon');
model.component('comp1').physics('dl').feature('porous1').feature('pm1').set('kappa', {'kappa' '0' '0' '0' 'kappa' '0' '0' '0' 'kappa'});
model.component('comp1').physics('dl').feature('gr1').set('GravityType', 'Elevation');
model.component('comp1').physics('dl').feature('gr1').set('useRref', true);
model.component('comp1').physics('dl').feature('gr1').set('rref', {'0' 'L/2' '0'});
model.component('comp1').physics('dl').feature('init1').set('InitType', 'H');
model.component('comp1').physics('dl').create('sym1', 'Symmetry', 1);
model.component('comp1').physics('dl').feature('sym1').selection.set([5]);
model.component('comp1').physics('dl').create('constr1', 'PointwiseConstraint', 0);
model.component('comp1').physics('dl').feature('constr1').selection.set([2]);
model.component('comp1').physics('dl').feature('constr1').set('constraintExpression', 'p0-p');
model.component('comp1').physics('tds').feature('porous1').feature('fluid1').set('u_src', 'fromCommonDef');
model.component('comp1').physics('tds').feature('porous1').feature('fluid1').set('DF_c', {'D_L' '0' '0' '0' 'D_L' '0' '0' '0' 'D_L'});
model.component('comp1').physics('tds').feature('porous1').feature('fluid1').set('FluidDiffusivityModelType', 'TortuosityModel');
model.component('comp1').physics('tds').feature('porous1').feature('pm1').set('poro_mat', 'userdef');
model.component('comp1').physics('tds').feature('porous1').feature('pm1').set('poro', 'epsilon');
model.component('comp1').physics('tds').feature('init1').setIndex('initc', 'c0', 0);
model.component('comp1').physics('tds').create('sym1', 'Symmetry', 1);
model.component('comp1').physics('tds').feature('sym1').selection.set([5]);
model.component('comp1').physics('tds').create('conc1', 'Concentration', 1);
model.component('comp1').physics('tds').feature('conc1').selection.set([2]);
model.component('comp1').physics('tds').feature('conc1').setIndex('species', true, 0);
model.component('comp1').physics('tds').feature('conc1').setIndex('c0', 'c0', 0);
model.component('comp1').physics('tds').create('conc2', 'Concentration', 1);
model.component('comp1').physics('tds').feature('conc2').setIndex('species', true, 0);
model.component('comp1').physics('tds').feature('conc2').setIndex('c0', 'c_s', 0);
model.component('comp1').physics('tds').feature('conc2').selection.set([4]);
model.component('comp1').physics('tds').create('ss1', 'SpeciesSource', 2);
model.component('comp1').physics('tds').feature('ss1').setIndex('S', 'A*exp(-((x-x0)^2 + (y-y0)^2)[1/m^2]/(2*sigma^2))/(365*24*60*60)', 0);
model.component('comp1').physics('tds').feature('ss1').selection.set([1]);

model.component('comp1').mesh('mesh1').automatic(false);
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 1);

model.study('std1').feature('time').set('tunit', 'a');
model.study('std1').feature('time').set('tlist', 'range(0,2,20)');

model.sol.create('sol1');

model.component('comp1').mesh('mesh1').stat.selection.geom(2);
model.component('comp1').mesh('mesh1').stat.selection.set([1]);
model.component('comp1').mesh('mesh1').stat.selection.geom(2);
model.component('comp1').mesh('mesh1').stat.selection.set([1]);

model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'time');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'time');
model.sol('sol1').create('t1', 'Time');
model.sol('sol1').feature('t1').set('tlist', 'range(0,2,20)');
model.sol('sol1').feature('t1').set('plot', 'off');
model.sol('sol1').feature('t1').set('plotgroup', 'Default');
model.sol('sol1').feature('t1').set('plotfreq', 'tout');
model.sol('sol1').feature('t1').set('probesel', 'all');
model.sol('sol1').feature('t1').set('probes', {});
model.sol('sol1').feature('t1').set('probefreq', 'tsteps');
model.sol('sol1').feature('t1').set('rtol', 0.005);
model.sol('sol1').feature('t1').set('atolglobalvaluemethod', 'factor');
model.sol('sol1').feature('t1').set('reacf', true);
model.sol('sol1').feature('t1').set('storeudot', true);
model.sol('sol1').feature('t1').set('endtimeinterpolation', true);
model.sol('sol1').feature('t1').set('maxorder', 2);
model.sol('sol1').feature('t1').set('stabcntrl', true);
model.sol('sol1').feature('t1').set('control', 'time');
model.sol('sol1').feature('t1').feature('aDef').set('cachepattern', true);
model.sol('sol1').feature('t1').create('seDef', 'Segregated');
model.sol('sol1').feature('t1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('t1').feature('fc1').set('jtech', 'once');
model.sol('sol1').feature('t1').feature('fc1').set('maxiter', 8);
model.sol('sol1').feature('t1').feature('fc1').set('stabacc', 'aacc');
model.sol('sol1').feature('t1').feature('fc1').set('aaccdim', 5);
model.sol('sol1').feature('t1').feature('fc1').set('aaccmix', 0.9);
model.sol('sol1').feature('t1').feature('fc1').set('aaccdelay', 1);
model.sol('sol1').feature('t1').feature('fc1').set('damp', 0.9);
model.sol('sol1').feature('t1').create('d1', 'Direct');
model.sol('sol1').feature('t1').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('t1').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('t1').feature('d1').label([native2unicode(hex2dec({'76' 'f4'}), 'unicode')  native2unicode(hex2dec({'63' 'a5'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ' (dl) (' native2unicode(hex2dec({'5d' 'f2'}), 'unicode')  native2unicode(hex2dec({'54' '08'}), 'unicode')  native2unicode(hex2dec({'5e' '76'}), 'unicode') ')']);
model.sol('sol1').feature('t1').create('i1', 'Iterative');
model.sol('sol1').feature('t1').feature('i1').set('linsolver', 'gmres');
model.sol('sol1').feature('t1').feature('i1').set('prefuntype', 'left');
model.sol('sol1').feature('t1').feature('i1').set('itrestart', 50);
model.sol('sol1').feature('t1').feature('i1').set('rhob', 400);
model.sol('sol1').feature('t1').feature('i1').set('maxlinit', 50);
model.sol('sol1').feature('t1').feature('i1').set('nlinnormuse', 'on');
model.sol('sol1').feature('t1').feature('i1').label(['AMG' native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' (tds)']);
model.sol('sol1').feature('t1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('prefun', 'saamg');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('mgcycle', 'v');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('maxcoarsedof', 50000);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('strconn', 0.01);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('nullspace', 'constant');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('usesmooth', false);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('saamgcompwise', true);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('loweramg', true);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').set('compactaggregation', false);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').create('sl1', 'SORLine');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('iter', 1);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('linerelax', 0.7);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('linealgorithm', 'mesh');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('linemethod', 'coupled');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('seconditer', 1);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('pr').feature('sl1').set('relax', 0.5);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').create('sl1', 'SORLine');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('iter', 1);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('linerelax', 0.7);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('linealgorithm', 'mesh');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('linemethod', 'coupled');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('seconditer', 1);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('po').feature('sl1').set('relax', 0.5);
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('t1').feature('i1').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('t1').create('i2', 'Iterative');
model.sol('sol1').feature('t1').feature('i2').set('linsolver', 'gmres');
model.sol('sol1').feature('t1').feature('i2').set('prefuntype', 'left');
model.sol('sol1').feature('t1').feature('i2').set('itrestart', 50);
model.sol('sol1').feature('t1').feature('i2').set('rhob', 400);
model.sol('sol1').feature('t1').feature('i2').set('maxlinit', 50);
model.sol('sol1').feature('t1').feature('i2').set('nlinnormuse', 'on');
model.sol('sol1').feature('t1').feature('i2').label(['AMG' native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ' (dl)']);
model.sol('sol1').feature('t1').feature('i2').create('mg1', 'Multigrid');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('prefun', 'saamg');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('mgcycle', 'v');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('maxcoarsedof', 50000);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('strconn', 0.01);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('nullspace', 'constant');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('usesmooth', false);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('saamgcompwise', false);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('loweramg', true);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').set('compactaggregation', false);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').create('sl1', 'SORLine');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('iter', 1);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('linerelax', 0.7);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('linealgorithm', 'mesh');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('linemethod', 'coupled');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('seconditer', 1);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('pr').feature('sl1').set('relax', 0.5);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').create('sl1', 'SORLine');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('iter', 1);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('linerelax', 0.7);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('linealgorithm', 'mesh');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('linemethod', 'coupled');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('seconditer', 1);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('po').feature('sl1').set('relax', 0.5);
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('t1').feature('i2').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('t1').feature('fc1').set('linsolver', 'd1');
model.sol('sol1').feature('t1').feature('fc1').set('jtech', 'once');
model.sol('sol1').feature('t1').feature('fc1').set('maxiter', 8);
model.sol('sol1').feature('t1').feature('fc1').set('stabacc', 'aacc');
model.sol('sol1').feature('t1').feature('fc1').set('aaccdim', 5);
model.sol('sol1').feature('t1').feature('fc1').set('aaccmix', 0.9);
model.sol('sol1').feature('t1').feature('fc1').set('aaccdelay', 1);
model.sol('sol1').feature('t1').feature('fc1').set('damp', 0.9);
model.sol('sol1').feature('t1').feature.remove('fcDef');
model.sol('sol1').feature('t1').feature.remove('seDef');
model.sol('sol1').attach('std1');
model.sol('sol1').runAll;

model.result.create('pg1', 'PlotGroup2D');
model.result('pg1').label([native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ' (dl)']);
model.result('pg1').set('titletype', 'custom');
model.result('pg1').set('data', 'dset1');
model.result('pg1').setIndex('looplevel', 11, 0);
model.result('pg1').set('defaultPlotID', 'PhysicsInterfaces_PorousMediaFlow/icom6/pdef1/pcond2/pg1');
model.result('pg1').feature.create('surf1', 'Surface');
model.result('pg1').feature('surf1').label([native2unicode(hex2dec({'88' '68'}), 'unicode')  native2unicode(hex2dec({'97' '62'}), 'unicode') ]);
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('smooth', 'internal');
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('data', 'parent');
model.result('pg1').feature.create('str1', 'Streamline');
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('posmethod', 'uniform');
model.result('pg1').feature('str1').set('pointtype', 'arrow');
model.result('pg1').feature('str1').set('arrowlength', 'logarithmic');
model.result('pg1').feature('str1').set('color', 'gray');
model.result('pg1').feature('str1').set('smooth', 'internal');
model.result('pg1').feature('str1').set('maxlen', Inf);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxlen', Inf);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxlen', Inf);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxlen', Inf);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxlen', Inf);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('data', 'parent');
model.result('pg1').feature('str1').selection.geom('geom1', 1);
model.result('pg1').feature('str1').selection.set([1 2 3 4 5]);
model.result.create('pg2', 'PlotGroup2D');
model.result('pg2').set('data', 'dset1');
model.result('pg2').setIndex('looplevel', 11, 0);
model.result('pg2').label([native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' (tds)']);
model.result('pg2').set('titletype', 'custom');
model.result('pg2').set('prefixintitle', '');
model.result('pg2').set('expressionintitle', false);
model.result('pg2').set('typeintitle', true);
model.result('pg2').create('surf1', 'Surface');
model.result('pg2').feature('surf1').set('expr', {'c'});
model.result('pg2').create('str1', 'Streamline');
model.result('pg2').feature('str1').set('expr', {'tds.tflux_cx' 'tds.tflux_cy'});
model.result('pg2').feature('str1').set('posmethod', 'uniform');
model.result('pg2').feature('str1').set('recover', 'pprint');
model.result('pg2').feature('str1').set('pointtype', 'arrow');
model.result('pg2').feature('str1').set('arrowlength', 'logarithmic');
model.result('pg2').feature('str1').set('color', 'gray');
model.result('pg1').run;
model.result.export.create('data1', 'Data');
model.result.export('data1').setIndex('expr', 'dl.u', 0);
model.result.export('data1').setIndex('unit', 'm/s', 0);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'60' '3b'}), 'unicode')  native2unicode(hex2dec({'8f' 'be'}), 'unicode')  native2unicode(hex2dec({'89' '7f'}), 'unicode')  native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'x ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 0);
model.result.export('data1').setIndex('expr', 'dl.v', 1);
model.result.export('data1').setIndex('unit', 'm/s', 1);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'60' '3b'}), 'unicode')  native2unicode(hex2dec({'8f' 'be'}), 'unicode')  native2unicode(hex2dec({'89' '7f'}), 'unicode')  native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 1);
model.result.export('data1').setIndex('expr', 'c', 2);
model.result.export('data1').setIndex('unit', 'mol/m^3', 2);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ], 2);
model.result.export('data1').setIndex('expr', 'A*exp(-((x-x0)^2 + (y-y0)^2)[1/m^2]/(2*sigma^2))/(365*24*60*60)', 3);
model.result.export('data1').setIndex('unit', '', 3);
model.result.export('data1').setIndex('descr', '', 3);

% model.result.export('data1').set('filename', 'E:\DATA\Elder\Untitled.csv');
% path_result=['data\' num2str(parm_NN) '.csv'];  %path_result
path_result=['data\data.csv'];  %path_result
model.result.export('data1').set('filename', path_result);

model.result.export('data1').set('location', 'regulargrid');
model.result.export('data1').set('regulargridx2', 128);
model.result.export('data1').set('regulargridy2', 128);
model.result.export('data1').run;

out = model;
