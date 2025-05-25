function out = TE_heat_v4(parm)

% pram
parm_NN=parm.NN;
parm_Em=parm.Em;
parm_f=parm.f;
parm_Sigma_Si_coef=parm.Sigma_Si_coef;
parm_theta=parm.theta;
parm_h_heat=parm.h_heat;
parm_Pho_Al=parm.Pho_Al;

parm_e_a=parm.e_a;
parm_e_b=parm.e_b;
parm_angle=parm.angle;

%
% TE_heat_v4.m
%
% Model exported on Apr 14 2025, 15:25 by COMSOL 6.2.0.290.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('E:\DATA\TE_heat');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('emw', 'ElectromagneticWaves', 'geom1');
model.component('comp1').physics.create('ht', 'HeatTransfer', 'geom1');

model.component('comp1').geom('geom1').lengthUnit('mm');
model.component('comp1').geom('geom1').create('sq1', 'Square');
model.component('comp1').geom('geom1').feature('sq1').set('size', 148);
model.component('comp1').geom('geom1').feature('sq1').set('base', 'corner');
model.component('comp1').geom('geom1').feature.duplicate('sq2', 'sq1');
model.component('comp1').geom('geom1').feature('sq2').set('size', 128);
model.component('comp1').geom('geom1').feature('sq1').set('base', 'center');
model.component('comp1').geom('geom1').feature('sq2').set('base', 'center');
model.component('comp1').geom('geom1').run('sq2');
model.component('comp1').geom('geom1').create('sq3', 'Square');
model.component('comp1').geom('geom1').feature('sq3').set('size', 10);
model.component('comp1').geom('geom1').feature('sq3').set('base', 'center');
model.component('comp1').geom('geom1').feature('sq3').set('pos', [-69 -69]);
model.component('comp1').geom('geom1').run('sq3');
model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size', [128 10]);
model.component('comp1').geom('geom1').feature('r1').set('base', 'center');
model.component('comp1').geom('geom1').feature('r1').set('pos', [0 69]);
model.component('comp1').geom('geom1').run('r1');
model.component('comp1').geom('geom1').create('sq4', 'Square');
model.component('comp1').geom('geom1').feature('sq4').set('size', 10);
model.component('comp1').geom('geom1').feature('sq4').set('base', 'center');
model.component('comp1').geom('geom1').feature('sq4').set('pos', [69 69]);
model.component('comp1').geom('geom1').run('sq4');
model.component('comp1').geom('geom1').feature.duplicate('sq5', 'sq4');
model.component('comp1').geom('geom1').feature('sq5').set('pos', [-69 -69]);
model.component('comp1').geom('geom1').feature.duplicate('sq6', 'sq5');
model.component('comp1').geom('geom1').feature('sq6').set('pos', [-69 69]);
model.component('comp1').geom('geom1').run('sq6');
model.component('comp1').geom('geom1').create('r2', 'Rectangle');
model.component('comp1').geom('geom1').feature('r2').set('size', [128 10]);
model.component('comp1').geom('geom1').feature('r2').set('base', 'center');
model.component('comp1').geom('geom1').feature('r2').set('pos', [0 -69]);
model.component('comp1').geom('geom1').feature.duplicate('r3', 'r2');
model.component('comp1').geom('geom1').feature('r3').set('size', [10 128]);
model.component('comp1').geom('geom1').feature('r3').set('pos', [69 0]);
model.component('comp1').geom('geom1').feature.duplicate('r4', 'r3');
model.component('comp1').geom('geom1').feature('r4').set('pos', [-69 0]);
model.component('comp1').geom('geom1').run('r4');
model.component('comp1').geom('geom1').create('e1', 'Ellipse');
model.component('comp1').geom('geom1').feature('e1').set('semiaxes', [20 1]);

model.param.set('Pho_Al', parm_Pho_Al);
model.param.descr('Pho_Al', '');
model.param.set('Em', parm_Em);
model.param.descr('Em', '');
model.param.set('theta', parm_theta);
model.param.descr('theta', '');
model.param.set('f', parm_f);
model.param.descr('f', '');
model.param.set('h_heat', parm_h_heat);
model.param.descr('h_heat', '');
model.param.set('Sigma_Si_coef', parm_Sigma_Si_coef);
model.param.descr('Sigma_Si_coef', '');
model.param.set('e_a', parm_e_a);
model.param.set('e_b', parm_e_b);
model.param.set('angle', parm_angle);

model.component('comp1').geom('geom1').feature('e1').set('semiaxes', {'e_a' 'e_b'});
model.component('comp1').geom('geom1').feature('e1').set('rot', 'angle');
model.component('comp1').geom('geom1').run('fin');

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat1').propertyGroup.create('RefractiveIndex', 'Refractive index');
model.component('comp1').material('mat1').label('Silicon');
model.component('comp1').material('mat1').set('family', 'custom');
model.component('comp1').material('mat1').set('customspecular', [0.7843137254901961 1 1]);
model.component('comp1').material('mat1').set('diffuse', 'custom');
model.component('comp1').material('mat1').set('customdiffuse', [0.6666666666666666 0.6666666666666666 0.7058823529411765]);
model.component('comp1').material('mat1').set('ambient', 'custom');
model.component('comp1').material('mat1').set('customambient', [0.6666666666666666 0.6666666666666666 0.7058823529411765]);
model.component('comp1').material('mat1').set('noise', true);
model.component('comp1').material('mat1').set('fresnel', 0.7);
model.component('comp1').material('mat1').set('roughness', 0.5);
model.component('comp1').material('mat1').set('diffusewrap', 0);
model.component('comp1').material('mat1').set('reflectance', 0);
model.component('comp1').material('mat1').propertyGroup('def').label('Basic');
model.component('comp1').material('mat1').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'1e-12[S/m]' '0' '0' '0' '1e-12[S/m]' '0' '0' '0' '1e-12[S/m]'});
model.component('comp1').material('mat1').propertyGroup('def').set('thermalexpansioncoefficient', {'2.6e-6[1/K]' '0' '0' '0' '2.6e-6[1/K]' '0' '0' '0' '2.6e-6[1/K]'});
model.component('comp1').material('mat1').propertyGroup('def').set('heatcapacity', '700[J/(kg*K)]');
model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', {'11.7' '0' '0' '0' '11.7' '0' '0' '0' '11.7'});
model.component('comp1').material('mat1').propertyGroup('def').set('density', '2329[kg/m^3]');
model.component('comp1').material('mat1').propertyGroup('def').set('thermalconductivity', {'130[W/(m*K)]' '0' '0' '0' '130[W/(m*K)]' '0' '0' '0' '130[W/(m*K)]'});
model.component('comp1').material('mat1').propertyGroup('Enu').label('Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat1').propertyGroup('Enu').set('E', '170[GPa]');
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', '0.28');
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').label('Refractive index');
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').set('n', {'3.48' '0' '0' '0' '3.48' '0' '0' '0' '3.48'});
model.component('comp1').material.create('mat2', 'Common');
model.component('comp1').material('mat2').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat2').label('Alumina');
model.component('comp1').material('mat2').set('family', 'aluminum');
model.component('comp1').material('mat2').propertyGroup('def').label('Basic');
model.component('comp1').material('mat2').propertyGroup('def').set('thermalexpansioncoefficient', {'8e-6[1/K]' '0' '0' '0' '8e-6[1/K]' '0' '0' '0' '8e-6[1/K]'});
model.component('comp1').material('mat2').propertyGroup('def').set('heatcapacity', '900[J/(kg*K)]');
model.component('comp1').material('mat2').propertyGroup('def').set('density', '3900[kg/m^3]');
model.component('comp1').material('mat2').propertyGroup('def').set('thermalconductivity', {'27[W/(m*K)]' '0' '0' '0' '27[W/(m*K)]' '0' '0' '0' '27[W/(m*K)]'});
model.component('comp1').material('mat2').propertyGroup('Enu').label('Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat2').propertyGroup('Enu').set('E', '300[GPa]');
model.component('comp1').material('mat2').propertyGroup('Enu').set('nu', '0.222');
model.component('comp1').material('mat1').selection.set([1 2 3 4 5 6 7 8 9 10]);
model.component('comp1').material('mat2').selection.set([1 2 3 4 5 6 7 8 9]);
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'an_Si(T)[S/m]'});
model.component('comp1').material('mat1').propertyGroup('def').set('thermalconductivity', {'70[W/(m*K)]'});
model.component('comp1').material('mat1').propertyGroup('def').func.create('an1', 'Analytic');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('funcname', 'an_Si');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('expr', '1.602*Sigma_Si_coef*exp(-1.12/(8.6173e-5*T))');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('args', 'T');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('fununit', 'S/m');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').setIndex('argunit', 'K', 0);
model.component('comp1').material('mat1').propertyGroup('def').func('an1').setIndex('plotargs', 290, 0, 1);
model.component('comp1').material('mat1').propertyGroup('def').func('an1').setIndex('plotargs', 350, 0, 2);
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'an_Si(T)'});
model.component('comp1').material('mat2').propertyGroup('def').set('relpermittivity', {'1'});
model.component('comp1').material('mat2').propertyGroup('def').set('relpermeability', {'1'});
model.component('comp1').material('mat2').propertyGroup('def').set('electricconductivity', {'1e-7'});
model.component('comp1').material('mat2').propertyGroup('def').set('thermalconductivity', {'Pho_Al[W/(m*K)]'});

model.component('comp1').coordSystem.create('pml1', 'PML');
model.component('comp1').coordSystem('pml1').selection.set([1 2 3 4 6 7 8 9]);

model.component('comp1').physics('emw').prop('BackgroundField').set('SolveFor', 'scatteredField');
model.component('comp1').physics('emw').prop('BackgroundField').set('Eb', {'0' '0' 'Em[V/m]*exp(j*emw.k0*(-x*cos(theta)-y*sin(theta)))'});
model.component('comp1').physics('emw').prop('ShapeProperty').set('order_electricfield', 7);
model.component('comp1').physics('emw').create('sctr1', 'Scattering', 1);
model.component('comp1').physics('emw').feature('sctr1').selection.set([1 2 3 5 7 9 14 16 21 22 23 24]);
model.component('comp1').physics('ht').selection.set([5 10]);
model.component('comp1').physics('ht').create('hf1', 'HeatFluxBoundary', 1);
model.component('comp1').physics('ht').feature('hf1').selection.set([10 11 13 17]);
model.component('comp1').physics('ht').feature('hf1').setIndex('materialType', 'from_mat', 0);
model.component('comp1').physics('ht').feature('hf1').set('HeatFluxType', 'ConvectiveHeatFlux');
model.component('comp1').physics('ht').feature('hf1').set('h', 'h_heat');

model.component('comp1').multiphysics.create('emh1', 'ElectromagneticHeating', 2);

model.component('comp1').mesh('mesh1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature.remove('size1');
model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTri');
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 2);
model.component('comp1').mesh('mesh1').run('ftri1');

model.study.create('std1');
model.study('std1').create('fstat', 'FrequencyStationary');
model.study('std1').feature('fstat').set('freq', '1000000');
model.study('std1').feature('fstat').set('solnum', 'auto');
model.study('std1').feature('fstat').set('notsolnum', 'auto');
model.study('std1').feature('fstat').set('outputmap', {});
model.study('std1').feature('fstat').set('ngenAUX', '1');
model.study('std1').feature('fstat').set('goalngenAUX', '1');
model.study('std1').feature('fstat').set('ngenAUX', '1');
model.study('std1').feature('fstat').set('goalngenAUX', '1');
model.study('std1').feature('fstat').setSolveFor('/physics/emw', true);
model.study('std1').feature('fstat').setSolveFor('/physics/ht', true);
model.study('std1').feature('fstat').setSolveFor('/multiphysics/emh1', true);
model.study('std1').feature('fstat').set('freq', 'f');

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'fstat');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'fstat');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', 0.001);
model.sol('sol1').feature('s1').feature('aDef').set('complexfun', true);
model.sol('sol1').feature('s1').feature('aDef').set('cachepattern', false);
model.sol('sol1').feature('s1').create('seDef', 'Segregated');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature('fc1').set('dtech', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('maxiter', 50);
model.sol('sol1').feature('s1').feature('fc1').set('initstep', 0.01);
model.sol('sol1').feature('s1').feature('fc1').set('minstep', 1.0E-6);
model.sol('sol1').feature('s1').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('s1').feature('d1').label([native2unicode(hex2dec({'5e' 'fa'}), 'unicode')  native2unicode(hex2dec({'8b' 'ae'}), 'unicode')  native2unicode(hex2dec({'76' '84'}), 'unicode')  native2unicode(hex2dec({'76' 'f4'}), 'unicode')  native2unicode(hex2dec({'63' 'a5'}), 'unicode')  native2unicode(hex2dec({'6c' '42'}), 'unicode')  native2unicode(hex2dec({'89' 'e3'}), 'unicode')  native2unicode(hex2dec({'56' '68'}), 'unicode') ' (emw) (' native2unicode(hex2dec({'5d' 'f2'}), 'unicode')  native2unicode(hex2dec({'54' '08'}), 'unicode')  native2unicode(hex2dec({'5e' '76'}), 'unicode') ')']);
model.sol('sol1').feature('s1').create('i1', 'Iterative');
model.sol('sol1').feature('s1').feature('i1').set('linsolver', 'gmres');
model.sol('sol1').feature('s1').feature('i1').set('prefuntype', 'left');
model.sol('sol1').feature('s1').feature('i1').set('itrestart', 50);
model.sol('sol1').feature('s1').feature('i1').set('rhob', 20);
model.sol('sol1').feature('s1').feature('i1').set('maxlinit', 10000);
model.sol('sol1').feature('s1').feature('i1').set('nlinnormuse', 'on');
model.sol('sol1').feature('s1').feature('i1').label(['AMG' native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'4f' '20'}), 'unicode')  native2unicode(hex2dec({'70' 'ed'}), 'unicode')  native2unicode(hex2dec({'53' 'd8'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ' (ht)']);
model.sol('sol1').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('prefun', 'saamg');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('mgcycle', 'v');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('maxcoarsedof', 50000);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('strconn', 0.01);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('nullspace', 'constant');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('usesmooth', false);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('saamgcompwise', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('loweramg', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('compactaggregation', false);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').create('so1', 'SOR');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('so1').set('iter', 2);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('so1').set('relax', 0.9);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').create('so1', 'SOR');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('so1').set('iter', 2);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('so1').set('relax', 0.9);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'd1');
model.sol('sol1').feature('s1').feature('fc1').set('dtech', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('maxiter', 50);
model.sol('sol1').feature('s1').feature('fc1').set('initstep', 0.01);
model.sol('sol1').feature('s1').feature('fc1').set('minstep', 1.0E-6);
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').feature('s1').feature.remove('seDef');
model.sol('sol1').attach('std1');
model.sol('sol1').runAll;

model.result.create('pg1', 'PlotGroup2D');
model.result('pg1').label([native2unicode(hex2dec({'75' '35'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode') ' (emw)']);
model.result('pg1').set('frametype', 'spatial');
model.result('pg1').set('showlegendsmaxmin', true);
model.result('pg1').set('data', 'dset1');
model.result('pg1').set('defaultPlotID', 'ElectromagneticWaves/phys1/pdef1/pcond2/pg1');
model.result('pg1').feature.create('surf1', 'Surface');
model.result('pg1').feature('surf1').label([native2unicode(hex2dec({'88' '68'}), 'unicode')  native2unicode(hex2dec({'97' '62'}), 'unicode') ]);
model.result('pg1').feature('surf1').set('smooth', 'internal');
model.result('pg1').feature('surf1').set('data', 'parent');
model.result.create('pg2', 'PlotGroup2D');
model.result('pg2').label([native2unicode(hex2dec({'6e' '29'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' (ht)']);
model.result('pg2').set('data', 'dset1');
model.result('pg2').set('defaultPlotID', 'ht/HT_PhysicsInterfaces/icom8/pdef1/pcond2/pcond4/pg2');
model.result('pg2').feature.create('surf1', 'Surface');
model.result('pg2').feature('surf1').set('showsolutionparams', 'on');
model.result('pg2').feature('surf1').set('solutionparams', 'parent');
model.result('pg2').feature('surf1').set('expr', 'T');
model.result('pg2').feature('surf1').set('colortable', 'HeatCameraLight');
model.result('pg2').feature('surf1').set('showsolutionparams', 'on');
model.result('pg2').feature('surf1').set('data', 'parent');
model.result('pg1').run;
model.result('pg2').run;
model.result.export.create('data1', 'Data');
model.result.export('data1').setIndex('expr', 'emw.Ez', 1);
path_result=['data\' num2str(parm_NN) '.csv'];  %path_result
model.result.export('data1').set('filename', path_result);
model.result.export('data1').set('location', 'regulargrid');
model.result.export('data1').set('regulargridx2', 148);
model.result.export('data1').set('regulargridy2', 148);
model.result.export('data1').run;

out = model;
