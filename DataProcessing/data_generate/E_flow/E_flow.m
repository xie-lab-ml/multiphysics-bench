function out = E_flow_v2(parm)

% pram
parm_NN=parm.NN;
parm_centerx=parm.centerx;
parm_centery=parm.centery;

parm_radius_a=parm.radius_a;
parm_radius_b=parm.radius_b;
parm_kappa0=parm.kappa0;
parm_kappa_x0=parm.kappa_x0;
parm_kappa_y0=parm.kappa_y0;
parm_kappa_sigma=parm.kappa_sigma;


%
% EOF_v2.m
%
% Model exported on Mar 29 2025, 21:51 by COMSOL 6.2.0.290.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('E:\DATA\EOF');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('ec', 'ConductiveMedia', 'geom1');
model.component('comp1').physics.create('g', 'GeneralFormPDE', {'u'});
model.component('comp1').physics('g').prop('EquationForm').set('form', 'Automatic');

model.study.create('std1');
model.study('std1').create('stat', 'Stationary');
model.study('std1').feature('stat').setSolveFor('/physics/ec', true);
model.study('std1').feature('stat').setSolveFor('/physics/g', true);

model.param.set('T', '298[K]');
model.param.descr('T', [native2unicode(hex2dec({'6e' '29'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ]);
model.param.set('eps_p', '0.6');
model.param.descr('eps_p', [native2unicode(hex2dec({'5b' '54'}), 'unicode')  native2unicode(hex2dec({'96' '99'}), 'unicode')  native2unicode(hex2dec({'73' '87'}), 'unicode') ]);
model.param.set('a', '10[um]');
model.param.descr('a', [native2unicode(hex2dec({'5e' '73'}), 'unicode')  native2unicode(hex2dec({'57' '47'}), 'unicode')  native2unicode(hex2dec({'5b' '54'}), 'unicode')  native2unicode(hex2dec({'96' '99'}), 'unicode')  native2unicode(hex2dec({'53' '4a'}), 'unicode')  native2unicode(hex2dec({'5f' '84'}), 'unicode') ]);
model.param.set('kappa0', parm_kappa0);
model.param.descr('kappa0', [native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'89' 'e3'}), 'unicode')  native2unicode(hex2dec({'8d' '28'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'5b' 'fc'}), 'unicode')  native2unicode(hex2dec({'73' '87'}), 'unicode') ]);
model.param.set('sigma', parm_kappa_sigma);
model.param.descr('sigma', '');
model.param.set('x0', parm_kappa_x0);
model.param.descr('x0', '');
model.param.set('y0', parm_kappa_y0);
model.param.descr('y0', '');
model.param.set('centerx', parm_centerx);
model.param.descr('centerx', '');
model.param.set('centery', parm_centery);
model.param.descr('centery', '');
model.param.set('radius_a', parm_radius_a);
model.param.descr('radius_a', '');
model.param.set('radius_b', parm_radius_b);
model.param.descr('radius_b', '');

model.param.set('L', '1.28[mm]');
model.param.descr('L', '');
model.param.set('V_anode', '50[V]');
model.param.descr('V_anode', [native2unicode(hex2dec({'96' '33'}), 'unicode')  native2unicode(hex2dec({'67' '81'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'4f' '4d'}), 'unicode') ]);
model.param.set('eps_w', '80.2*epsilon0_const');
model.param.descr('eps_w', [native2unicode(hex2dec({'4e' 'cb'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'5e' '38'}), 'unicode')  native2unicode(hex2dec({'65' '70'}), 'unicode') ]);
model.param.set('eta', '1e-3[Pa*s]');
model.param.descr('eta', [native2unicode(hex2dec({'52' 'a8'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'9e' 'cf'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ]);
model.param.set('zeta', '-0.1[V]');
model.param.descr('zeta', ['Zeta ' native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'4f' '4d'}), 'unicode') ]);
model.param.set('p1', '0.01*1.013e5[Pa]');
model.param.descr('p1', [native2unicode(hex2dec({'51' '65'}), 'unicode')  native2unicode(hex2dec({'53' 'e3'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ]);
model.param.set('k_p', 'eps_p*a^2/(8*eta)');
model.param.descr('k_p', [native2unicode(hex2dec({'52' '4d'}), 'unicode')  native2unicode(hex2dec({'56' 'e0'}), 'unicode')  native2unicode(hex2dec({'5b' '50'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'52' 'a8'}), 'unicode') '-' native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode') ]);
model.param.set('k_V', 'eps_p*eps_w*zeta/eta');
model.param.descr('k_V', [native2unicode(hex2dec({'52' '4d'}), 'unicode')  native2unicode(hex2dec({'56' 'e0'}), 'unicode')  native2unicode(hex2dec({'5b' '50'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'52' 'a8'}), 'unicode') '-' native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode') ]);
model.param.set('D', '1e-9[m^2/s]');
model.param.descr('D', [native2unicode(hex2dec({'79' '3a'}), 'unicode')  native2unicode(hex2dec({'8e' '2a'}), 'unicode')  native2unicode(hex2dec({'52' '42'}), 'unicode')  native2unicode(hex2dec({'62' '69'}), 'unicode')  native2unicode(hex2dec({'65' '63'}), 'unicode')  native2unicode(hex2dec({'7c' 'fb'}), 'unicode')  native2unicode(hex2dec({'65' '70'}), 'unicode') ]);
model.param.set('zn', '1');
model.param.descr('zn', [native2unicode(hex2dec({'79' '3a'}), 'unicode')  native2unicode(hex2dec({'8e' '2a'}), 'unicode')  native2unicode(hex2dec({'79' 'bb'}), 'unicode')  native2unicode(hex2dec({'5b' '50'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'83' '77'}), 'unicode')  native2unicode(hex2dec({'65' '70'}), 'unicode') ]);
model.param.set('ctop', '1[mmol/m^3]');
model.param.descr('ctop', [native2unicode(hex2dec({'79' '3a'}), 'unicode')  native2unicode(hex2dec({'8e' '2a'}), 'unicode')  native2unicode(hex2dec({'52' '42'}), 'unicode')  native2unicode(hex2dec({'67' '00'}), 'unicode')  native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'52' '1d'}), 'unicode')  native2unicode(hex2dec({'59' 'cb'}), 'unicode')  native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ]);
model.param.set('p_w', '0.2[mm]');
model.param.descr('p_w', [native2unicode(hex2dec({'79' '3a'}), 'unicode')  native2unicode(hex2dec({'8e' '2a'}), 'unicode')  native2unicode(hex2dec({'52' '42'}), 'unicode')  native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'5e' '03'}), 'unicode')  native2unicode(hex2dec({'66' 'f2'}), 'unicode')  native2unicode(hex2dec({'7e' 'bf'}), 'unicode')  native2unicode(hex2dec({'5c' 'f0'}), 'unicode')  native2unicode(hex2dec({'50' '3c'}), 'unicode')  native2unicode(hex2dec({'5b' 'bd'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ]);
model.param.set('x_m', '3[mm]');
model.param.descr('x_m', [native2unicode(hex2dec({'51' 'e0'}), 'unicode')  native2unicode(hex2dec({'4f' '55'}), 'unicode')  native2unicode(hex2dec({'51' '85'}), 'unicode')  native2unicode(hex2dec({'76' '84'}), 'unicode')  native2unicode(hex2dec({'79' '3a'}), 'unicode')  native2unicode(hex2dec({'8e' '2a'}), 'unicode')  native2unicode(hex2dec({'52' '42'}), 'unicode')  native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'5e' '03'}), 'unicode')  native2unicode(hex2dec({'66' 'f2'}), 'unicode')  native2unicode(hex2dec({'7e' 'bf'}), 'unicode')  native2unicode(hex2dec({'5c' 'f0'}), 'unicode')  native2unicode(hex2dec({'50' '3c'}), 'unicode')  native2unicode(hex2dec({'4f' '4d'}), 'unicode')  native2unicode(hex2dec({'7f' '6e'}), 'unicode') ]);

model.component('comp1').geom('geom1').lengthUnit('mm');

model.component('comp1').variable.create('var1');

model.component('comp1').geom('geom1').run;

model.component('comp1').variable('var1').set('u_p', '-k_p*px');
model.component('comp1').variable('var1').descr('u_p', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'x ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('v_p', '-k_p*py');
model.component('comp1').variable('var1').descr('v_p', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('U_p', 'sqrt(u_p^2+v_p^2)');
model.component('comp1').variable('var1').descr('U_p', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'5c' '0f'}), 'unicode') ]);
model.component('comp1').variable('var1').set('u_eo', 'k_V*Vx');
model.component('comp1').variable('var1').descr('u_eo', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'x ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('v_eo', 'k_V*Vy');
model.component('comp1').variable('var1').descr('v_eo', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('U_eo', 'sqrt(u_eo^2+v_eo^2)');
model.component('comp1').variable('var1').descr('U_eo', [native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'52' 'a8'}), 'unicode') '-' native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'98' '79'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'5c' '0f'}), 'unicode') ]);
model.component('comp1').variable('var1').set('u_flow', 'u_p+u_eo');
model.component('comp1').variable('var1').descr('u_flow', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' x ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('v_flow', 'v_p+v_eo');
model.component('comp1').variable('var1').descr('v_flow', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ]);
model.component('comp1').variable('var1').set('U_flow', 'sqrt(u_flow^2+v_flow^2)');
model.component('comp1').variable('var1').descr('U_flow', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'5c' '0f'}), 'unicode') ]);
model.component('comp1').variable('var1').set('c_init', 'ctop*exp(-0.5*((x-x_m)/p_w)^2)');
model.component('comp1').variable('var1').descr('c_init', [native2unicode(hex2dec({'52' '1d'}), 'unicode')  native2unicode(hex2dec({'59' 'cb'}), 'unicode')  native2unicode(hex2dec({'6d' '53'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode')  native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'5e' '03'}), 'unicode') ]);
model.component('comp1').variable('var1').set('kappa', 'kappa0*exp(-((x - x0)^2 + (y - y0)^2)/(2*sigma^2))');
model.component('comp1').variable('var1').descr('kappa', '');

model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size', {'L' 'L'});
model.component('comp1').geom('geom1').feature('r1').set('base', 'center');
model.component('comp1').geom('geom1').run('r1');
model.component('comp1').geom('geom1').create('e1', 'Ellipse');
model.component('comp1').geom('geom1').feature('e1').set('semiaxes', {'radius_a' 'radius_b'});
model.component('comp1').geom('geom1').feature('e1').set('pos', {'centerx' 'centery'});
model.component('comp1').geom('geom1').run('e1');
model.component('comp1').geom('geom1').create('dif1', 'Difference');
model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'r1'});
model.component('comp1').geom('geom1').feature('dif1').selection('input2').set({'e1'});
model.component('comp1').geom('geom1').run;

model.component('comp1').physics('ec').feature('cucn1').set('sigma_mat', 'userdef');
model.component('comp1').physics('ec').feature('cucn1').set('sigma', {'kappa' '0' '0' '0' 'kappa' '0' '0' '0' 'kappa'});
model.component('comp1').physics('ec').feature('cucn1').set('epsilonr_mat', 'userdef');
model.component('comp1').physics('ec').create('pot1', 'ElectricPotential', 1);
model.component('comp1').physics('ec').feature('pot1').selection.set([5 6 7 8]);
model.component('comp1').physics('ec').create('pot2', 'ElectricPotential', 1);
model.component('comp1').physics('ec').feature('pot2').selection.set([1 4]);
model.component('comp1').physics('ec').feature('pot2').set('V0', 'V_anode');
model.component('comp1').physics('g').label([native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ]);
model.component('comp1').physics('g').prop('Units').set('DependentVariableQuantity', 'pressure');
model.component('comp1').physics('g').prop('Units').setIndex('CustomSourceTermUnit', '1/s', 0, 0);
model.component('comp1').physics('g').field('dimensionless').field('p');
model.component('comp1').physics('g').field('dimensionless').component(1, 'p');
model.component('comp1').physics('g').feature('gfeq1').setIndex('Ga', {'u_flow' '-uy'}, 0);
model.component('comp1').physics('g').feature('gfeq1').setIndex('Ga', {'u_flow' 'v_flow'}, 0);
model.component('comp1').physics('g').feature('gfeq1').setIndex('f', 0, 0);
model.component('comp1').physics('g').feature('gfeq1').setIndex('da', 0, 0);
model.component('comp1').physics('g').create('dir1', 'DirichletBoundary', 1);
model.component('comp1').physics('g').feature('dir1').label([native2unicode(hex2dec({'51' '65'}), 'unicode')  native2unicode(hex2dec({'53' 'e3'}), 'unicode') ' - p=0']);
model.component('comp1').physics('g').feature('dir1').selection.set([1]);
model.component('comp1').physics('g').create('dir2', 'DirichletBoundary', 1);
model.component('comp1').physics('g').feature('dir2').label([native2unicode(hex2dec({'51' 'fa'}), 'unicode')  native2unicode(hex2dec({'53' 'e3'}), 'unicode') ' - p=p1']);
model.component('comp1').physics('g').feature('dir2').selection.set([4]);
model.component('comp1').physics('g').feature('dir2').setIndex('r', 'p1', 0);

model.component('comp1').mesh('mesh1').autoMeshSize(1);

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'stat');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'stat');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').create('seDef', 'Segregated');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'dDef');
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').feature('s1').feature.remove('seDef');
model.sol('sol1').attach('std1');
model.sol('sol1').runAll;

model.result.create('pg1', 'PlotGroup2D');
model.result('pg1').label([native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'52' 'bf'}), 'unicode') ' (ec)']);
model.result('pg1').set('frametype', 'spatial');
model.result('pg1').set('showlegendsmaxmin', true);
model.result('pg1').set('data', 'dset1');
model.result('pg1').set('defaultPlotID', 'InterfaceComponents/PlotDefaults/icom2/pdef1/pcond2/pcond2/pg1');
model.result('pg1').feature.create('surf1', 'Surface');
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('solutionparams', 'parent');
model.result('pg1').feature('surf1').set('colortable', 'Dipole');
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('data', 'parent');
model.result('pg1').feature.create('str1', 'Streamline');
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('solutionparams', 'parent');
model.result('pg1').feature('str1').set('expr', {'ec.Ex' 'ec.Ey'});
model.result('pg1').feature('str1').set('titletype', 'none');
model.result('pg1').feature('str1').set('posmethod', 'uniform');
model.result('pg1').feature('str1').set('udist', 0.02);
model.result('pg1').feature('str1').set('maxlen', 0.4);
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('inheritcolor', false);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('showsolutionparams', 'on');
model.result('pg1').feature('str1').set('maxtime', Inf);
model.result('pg1').feature('str1').set('data', 'parent');
model.result('pg1').feature('str1').selection.geom('geom1', 1);
model.result('pg1').feature('str1').selection.set([1 2 3 4 5 6 7 8]);
model.result('pg1').feature('str1').set('inheritplot', 'surf1');
model.result('pg1').feature('str1').feature.create('col1', 'Color');
model.result('pg1').feature('str1').feature('col1').set('colortable', 'DipoleDark');
model.result('pg1').feature('str1').feature('col1').set('colorlegend', false);
model.result('pg1').feature('str1').feature.create('filt1', 'Filter');
model.result('pg1').feature('str1').feature('filt1').set('expr', '!isScalingSystemDomain');
model.result.create('pg2', 'PlotGroup2D');
model.result('pg2').label([native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode')  native2unicode(hex2dec({'6a' '21'}), 'unicode') ' (ec)']);
model.result('pg2').set('frametype', 'spatial');
model.result('pg2').set('showlegendsmaxmin', true);
model.result('pg2').set('data', 'dset1');
model.result('pg2').set('defaultPlotID', 'InterfaceComponents/PlotDefaults/icom3/pdef1/pcond2/pcond2/pg1');
model.result('pg2').feature.create('surf1', 'Surface');
model.result('pg2').feature('surf1').set('showsolutionparams', 'on');
model.result('pg2').feature('surf1').set('solutionparams', 'parent');
model.result('pg2').feature('surf1').set('expr', 'ec.normE');
model.result('pg2').feature('surf1').set('colortable', 'Prism');
model.result('pg2').feature('surf1').set('colortabletrans', 'nonlinear');
model.result('pg2').feature('surf1').set('colorcalibration', -0.8);
model.result('pg2').feature('surf1').set('showsolutionparams', 'on');
model.result('pg2').feature('surf1').set('data', 'parent');
model.result('pg2').feature.create('str1', 'Streamline');
model.result('pg2').feature('str1').set('showsolutionparams', 'on');
model.result('pg2').feature('str1').set('solutionparams', 'parent');
model.result('pg2').feature('str1').set('expr', {'ec.Ex' 'ec.Ey'});
model.result('pg2').feature('str1').set('titletype', 'none');
model.result('pg2').feature('str1').set('posmethod', 'uniform');
model.result('pg2').feature('str1').set('udist', 0.02);
model.result('pg2').feature('str1').set('maxlen', 0.4);
model.result('pg2').feature('str1').set('maxtime', Inf);
model.result('pg2').feature('str1').set('inheritcolor', false);
model.result('pg2').feature('str1').set('showsolutionparams', 'on');
model.result('pg2').feature('str1').set('maxtime', Inf);
model.result('pg2').feature('str1').set('showsolutionparams', 'on');
model.result('pg2').feature('str1').set('maxtime', Inf);
model.result('pg2').feature('str1').set('showsolutionparams', 'on');
model.result('pg2').feature('str1').set('maxtime', Inf);
model.result('pg2').feature('str1').set('showsolutionparams', 'on');
model.result('pg2').feature('str1').set('maxtime', Inf);
model.result('pg2').feature('str1').set('data', 'parent');
model.result('pg2').feature('str1').selection.geom('geom1', 1);
model.result('pg2').feature('str1').selection.set([1 2 3 4 5 6 7 8]);
model.result('pg2').feature('str1').set('inheritplot', 'surf1');
model.result('pg2').feature('str1').feature.create('col1', 'Color');
model.result('pg2').feature('str1').feature('col1').set('expr', 'ec.normE');
model.result('pg2').feature('str1').feature('col1').set('colortable', 'PrismDark');
model.result('pg2').feature('str1').feature('col1').set('colorlegend', false);
model.result('pg2').feature('str1').feature('col1').set('colortabletrans', 'nonlinear');
model.result('pg2').feature('str1').feature('col1').set('colorcalibration', -0.8);
model.result('pg2').feature('str1').feature.create('filt1', 'Filter');
model.result('pg2').feature('str1').feature('filt1').set('expr', '!isScalingSystemDomain');
model.result.create('pg3', 'PlotGroup2D');
model.result('pg3').set('data', 'dset1');
model.result('pg3').create('surf1', 'Surface');
model.result('pg3').label([native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'6e' '17'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ]);
model.result('pg3').feature('surf1').set('expr', 'p');
model.result('pg1').run;
model.result.export.create('data1', 'Data');
model.result.export('data1').setIndex('expr', 'V', 0);
model.result.export('data1').setIndex('unit', 'V', 0);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'52' 'bf'}), 'unicode') ], 0);
model.result.export('data1').setIndex('expr', 'u_flow', 1);
model.result.export('data1').setIndex('unit', 'm/s', 1);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' x ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 1);
model.result.export('data1').setIndex('expr', 'v_flow', 2);
model.result.export('data1').setIndex('unit', 'm/s', 2);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 2);
model.result.export('data1').setIndex('expr', 'kappa', 3);
model.result.export('data1').setIndex('unit', 'S/m', 3);
model.result.export('data1').setIndex('descr', '', 3);

% path_result=['E:\DATA\EOF\data\' num2str(parm_NN) '.csv'];  %path_result
path_result=['data\' num2str(parm_NN) '.csv'];  %path_result
model.result.export('data1').set('filename', path_result);

model.result.export('data1').set('location', 'regulargrid');
model.result.export('data1').set('regulargridx2', 128);
model.result.export('data1').set('regulargridy2', 128);
model.result.export('data1').run;

out = model;
