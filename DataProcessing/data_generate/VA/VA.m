function out = VA_v1(parm)
parm_NN=parm.NN;
parm_rho_A=parm.rho_A;  
parm_rho_x0=parm.rho_x0; 
parm_rho_y0=parm.rho_y0; 
parm_rho_sigma=parm.rho_sigma; 



%
% VA_v1.m
%
% Model exported on Apr 4 2025, 01:27 by COMSOL 6.2.0.290.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('D:\DATA\VA');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('acpr', 'PressureAcoustics', 'geom1');
model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');

model.study.create('std1');
model.study('std1').create('freq', 'Frequency');
model.study('std1').feature('freq').setSolveFor('/physics/acpr', true);
model.study('std1').feature('freq').setSolveFor('/physics/solid', true);

model.param.set('f', '50[kHz]');
model.param.descr('f', '');
model.param.set('phi', '-pi/6 [rad]');
model.param.descr('phi', '');
model.param.set('k1', 'cos(phi)');
model.param.descr('k1', '');
model.param.set('k2', 'sin(phi)');
model.param.descr('k2', '');
model.param.set('sigma', parm_rho_sigma);
model.param.descr('sigma', '');
model.param.set('x0', parm_rho_x0);
model.param.descr('x0', '');
model.param.set('y0', parm_rho_y0);
model.param.descr('y0', '');
model.param.set('A', parm_rho_A);
model.param.descr('A', '');
model.param.set('lambda', '2*pi*20');
model.param.descr('lambda', '');

model.component('comp1').variable.create('var1');

model.component('comp1').geom('geom1').run;

model.component('comp1').variable('var1').set('rho_water', '10 + A*exp(-((x-x0)^2 + (y-y0)^2)[1/mm^2]/(2*sigma^2))');
model.component('comp1').variable('var1').descr('rho_water', '');

model.component('comp1').geom('geom1').create('sq1', 'Square');
model.component('comp1').geom('geom1').lengthUnit('mm');
model.component('comp1').geom('geom1').feature('sq1').set('base', 'center');
model.component('comp1').geom('geom1').feature('sq1').set('size', 40);
model.component('comp1').geom('geom1').run('sq1');
model.component('comp1').geom('geom1').create('c1', 'Circle');
model.component('comp1').geom('geom1').feature('c1').set('r', 5);
model.component('comp1').geom('geom1').run;

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat1').label('Aluminum 3003-H18');
model.component('comp1').material('mat1').set('family', 'aluminum');
model.component('comp1').material('mat1').propertyGroup('def').label('Basic');
model.component('comp1').material('mat1').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'2.326e7[S/m]' '0' '0' '0' '2.326e7[S/m]' '0' '0' '0' '2.326e7[S/m]'});
model.component('comp1').material('mat1').propertyGroup('def').set('thermalexpansioncoefficient', {'23.2e-6[1/K]' '0' '0' '0' '23.2e-6[1/K]' '0' '0' '0' '23.2e-6[1/K]'});
model.component('comp1').material('mat1').propertyGroup('def').set('heatcapacity', '893[J/(kg*K)]');
model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').set('density', '2730[kg/m^3]');
model.component('comp1').material('mat1').propertyGroup('def').set('thermalconductivity', {'155[W/(m*K)]' '0' '0' '0' '155[W/(m*K)]' '0' '0' '0' '155[W/(m*K)]'});
model.component('comp1').material('mat1').propertyGroup('Enu').label('Young''s modulus and Poisson''s ratio');
model.component('comp1').material('mat1').propertyGroup('Enu').set('E', '69[GPa]');
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', '0.33');
model.component('comp1').material.create('mat2', 'Common');
model.component('comp1').material('mat2').propertyGroup('def').func.create('eta', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('Cp', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('rho', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('k', 'Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func.create('cs', 'Interpolation');
model.component('comp1').material('mat2').propertyGroup('def').func.create('an1', 'Analytic');
model.component('comp1').material('mat2').propertyGroup('def').func.create('an2', 'Analytic');
model.component('comp1').material('mat2').propertyGroup('def').func.create('an3', 'Analytic');
model.component('comp1').material('mat2').label('Water, liquid');
model.component('comp1').material('mat2').set('family', 'water');
model.component('comp1').material('mat2').propertyGroup('def').label('Basic');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').label('Piecewise');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('pieces', {'273.15' '413.15' '1.3799566804-0.021224019151*T^1+1.3604562827E-4*T^2-4.6454090319E-7*T^3+8.9042735735E-10*T^4-9.0790692686E-13*T^5+3.8457331488E-16*T^6'; '413.15' '553.75' '0.00401235783-2.10746715E-5*T^1+3.85772275E-8*T^2-2.39730284E-11*T^3'});
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('argunit', 'K');
model.component('comp1').material('mat2').propertyGroup('def').func('eta').set('fununit', 'Pa*s');
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').label('Piecewise 2');
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('pieces', {'273.15' '553.75' '12010.1471-80.4072879*T^1+0.309866854*T^2-5.38186884E-4*T^3+3.62536437E-7*T^4'});
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('argunit', 'K');
model.component('comp1').material('mat2').propertyGroup('def').func('Cp').set('fununit', 'J/(kg*K)');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').label('Piecewise 3');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('smooth', 'contd1');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('pieces', {'273.15' '293.15' '0.000063092789034*T^3-0.060367639882855*T^2+18.9229382407066*T-950.704055329848'; '293.15' '373.15' '0.000010335053319*T^3-0.013395065634452*T^2+4.969288832655160*T+432.257114008512'});
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('argunit', 'K');
model.component('comp1').material('mat2').propertyGroup('def').func('rho').set('fununit', 'kg/m^3');
model.component('comp1').material('mat2').propertyGroup('def').func('k').label('Piecewise 4');
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('arg', 'T');
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('pieces', {'273.15' '553.75' '-0.869083936+0.00894880345*T^1-1.58366345E-5*T^2+7.97543259E-9*T^3'});
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('argunit', 'K');
model.component('comp1').material('mat2').propertyGroup('def').func('k').set('fununit', 'W/(m*K)');
model.component('comp1').material('mat2').propertyGroup('def').func('cs').label('Interpolation');
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('table', {'273' '1403';  ...
'278' '1427';  ...
'283' '1447';  ...
'293' '1481';  ...
'303' '1507';  ...
'313' '1526';  ...
'323' '1541';  ...
'333' '1552';  ...
'343' '1555';  ...
'353' '1555';  ...
'363' '1550';  ...
'373' '1543'});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('interp', 'piecewisecubic');
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('fununit', {'m/s'});
model.component('comp1').material('mat2').propertyGroup('def').func('cs').set('argunit', {'K'});
model.component('comp1').material('mat2').propertyGroup('def').func('an1').label('Analytic 1');
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('funcname', 'alpha_p');
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('expr', '-1/rho(T)*d(rho(T),T)');
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('args', {'T'});
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('fununit', '1/K');
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('argunit', {'K'});
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('plotfixedvalue', {'273.15'});
model.component('comp1').material('mat2').propertyGroup('def').func('an1').set('plotargs', {'T' '273.15' '373.15'});
model.component('comp1').material('mat2').propertyGroup('def').func('an2').label('Analytic 2');
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('funcname', 'gamma_w');
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('expr', '1+(T/Cp(T))*(alpha_p(T)*cs(T))^2');
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('args', {'T'});
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('fununit', '1');
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('argunit', {'K'});
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('plotfixedvalue', {'273.15'});
model.component('comp1').material('mat2').propertyGroup('def').func('an2').set('plotargs', {'T' '273.15' '373.15'});
model.component('comp1').material('mat2').propertyGroup('def').func('an3').label('Analytic 3');
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('funcname', 'muB');
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('expr', '2.79*eta(T)');
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('args', {'T'});
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('fununit', 'Pa*s');
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('argunit', {'K'});
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('plotfixedvalue', {'273.15'});
model.component('comp1').material('mat2').propertyGroup('def').func('an3').set('plotargs', {'T' '273.15' '553.75'});
model.component('comp1').material('mat2').propertyGroup('def').set('thermalexpansioncoefficient', '');
model.component('comp1').material('mat2').propertyGroup('def').set('bulkviscosity', '');
model.component('comp1').material('mat2').propertyGroup('def').set('thermalexpansioncoefficient', {'alpha_p(T)' '0' '0' '0' 'alpha_p(T)' '0' '0' '0' 'alpha_p(T)'});
model.component('comp1').material('mat2').propertyGroup('def').set('bulkviscosity', 'muB(T)');
model.component('comp1').material('mat2').propertyGroup('def').set('dynamicviscosity', 'eta(T)');
model.component('comp1').material('mat2').propertyGroup('def').set('ratioofspecificheat', 'gamma_w(T)');
model.component('comp1').material('mat2').propertyGroup('def').set('electricconductivity', {'5.5e-6[S/m]' '0' '0' '0' '5.5e-6[S/m]' '0' '0' '0' '5.5e-6[S/m]'});
model.component('comp1').material('mat2').propertyGroup('def').set('heatcapacity', 'Cp(T)');
model.component('comp1').material('mat2').propertyGroup('def').set('density', 'rho(T)');
model.component('comp1').material('mat2').propertyGroup('def').set('thermalconductivity', {'k(T)' '0' '0' '0' 'k(T)' '0' '0' '0' 'k(T)'});
model.component('comp1').material('mat2').propertyGroup('def').set('soundspeed', 'cs(T)');
model.component('comp1').material('mat2').propertyGroup('def').addInput('temperature');
model.component('comp1').material('mat1').selection.set([1]);
model.component('comp1').material('mat2').selection.set([1]);
model.component('comp1').material('mat1').selection.set([1 2]);
model.component('comp1').material('mat2').propertyGroup('def').set('density', {'rho_water'});

model.component('comp1').physics('acpr').selection.set([1]);
model.component('comp1').physics('acpr').create('cwr1', 'CylindricalWaveRadiation', 1);
model.component('comp1').physics('acpr').feature('cwr1').selection.set([1 2 3 4]);
model.component('comp1').physics('acpr').feature('cwr1').create('ipf1', 'IncidentPressureField', 1);
model.component('comp1').physics('acpr').feature('cwr1').selection.set([1 2 3 4 5 6 7 8]);
model.component('comp1').physics('acpr').feature('cwr1').feature('ipf1').set('pamp', 1);
model.component('comp1').physics('acpr').feature('cwr1').feature('ipf1').set('c_mat', 'from_mat');
model.component('comp1').physics('acpr').feature('cwr1').feature('ipf1').set('dir', {'k1' 'k2' '0'});
model.component('comp1').physics('acpr').feature('cwr1').feature('ipf1').set('PressureFieldMaterial', 'mat2');
model.component('comp1').physics('solid').selection.set([2]);

model.component('comp1').multiphysics.create('asb1', 'AcousticStructureBoundary', 1);
model.component('comp1').multiphysics('asb1').selection.set([5 6 7 8]);

model.component('comp1').physics('acpr').prop('MeshControl').set('SizeControlParameter', 'Frequency');
model.component('comp1').physics('acpr').prop('MeshControl').set('ElementsPerWavelength', 'UserDefined');
model.component('comp1').physics('acpr').prop('MeshControl').set('nperlambda', 10);
model.component('comp1').physics('acpr').prop('ReferencePressure').set('ReferenceType', 'ReferencePressureWater');
model.study('std1').feature('freq').set('plist', 'f');

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'freq');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'freq');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', 0.001);
model.sol('sol1').feature('s1').create('p1', 'Parametric');
model.sol('sol1').feature('s1').feature.remove('pDef');
model.sol('sol1').feature('s1').feature('p1').set('pname', {'freq'});
model.sol('sol1').feature('s1').feature('p1').set('plistarr', {'f'});
model.sol('sol1').feature('s1').feature('p1').set('punit', {'Hz'});
model.sol('sol1').feature('s1').feature('p1').set('pcontinuationmode', 'no');
model.sol('sol1').feature('s1').feature('p1').set('preusesol', 'no');
model.sol('sol1').feature('s1').feature('p1').set('pdistrib', 'off');
model.sol('sol1').feature('s1').feature('p1').set('plot', 'off');
model.sol('sol1').feature('s1').feature('p1').set('plotgroup', 'Default');
model.sol('sol1').feature('s1').feature('p1').set('probesel', 'all');
model.sol('sol1').feature('s1').feature('p1').set('probes', {});
model.sol('sol1').feature('s1').feature('p1').set('control', 'freq');
model.sol('sol1').feature('s1').set('linpmethod', 'sol');
model.sol('sol1').feature('s1').set('linpsol', 'zero');
model.sol('sol1').feature('s1').set('control', 'freq');
model.sol('sol1').feature('s1').feature('aDef').set('complexfun', true);
model.sol('sol1').feature('s1').feature('aDef').set('cachepattern', true);
model.sol('sol1').feature('s1').feature('aDef').set('matherr', true);
model.sol('sol1').feature('s1').feature('aDef').set('blocksizeactive', false);
model.sol('sol1').feature('s1').create('seDef', 'Segregated');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'dDef');
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').feature('s1').feature.remove('seDef');
model.sol('sol1').attach('std1');
model.sol('sol1').runAll;

model.result.create('pg1', 'PlotGroup2D');
model.result('pg1').set('data', 'dset1');
model.result('pg1').setIndex('looplevel', 1, 0);
model.result('pg1').create('surf1', 'Surface');
model.result('pg1').feature('surf1').set('expr', {'acpr.p_t'});
model.result('pg1').feature('surf1').set('colortable', 'Wave');
model.result('pg1').feature('surf1').set('colorscalemode', 'linearsymmetric');
model.result('pg1').set('showlegendsunit', true);
model.result('pg1').label([native2unicode(hex2dec({'58' 'f0'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode') ' (acpr)']);
model.result.create('pg2', 'PlotGroup2D');
model.result('pg2').set('data', 'dset1');
model.result('pg2').setIndex('looplevel', 1, 0);
model.result('pg2').create('surf1', 'Surface');
model.result('pg2').feature('surf1').set('expr', {'acpr.Lp_t'});
model.result('pg2').feature('surf1').set('colortable', 'Rainbow');
model.result('pg2').feature('surf1').set('colorscalemode', 'linear');
model.result('pg2').set('showlegendsunit', true);
model.result('pg2').label([native2unicode(hex2dec({'58' 'f0'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'7e' 'a7'}), 'unicode') ' (acpr)']);
model.result.create('pg3', 'PlotGroup2D');
model.result('pg3').set('data', 'dset1');
model.result('pg3').setIndex('looplevel', 1, 0);
model.result('pg3').set('defaultPlotID', 'stress');
model.result('pg3').label([native2unicode(hex2dec({'5e' '94'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ' (solid)']);
model.result('pg3').set('frametype', 'spatial');
model.result('pg3').create('surf1', 'Surface');
model.result('pg3').feature('surf1').set('expr', {'solid.misesGp_peak'});
model.result('pg3').feature('surf1').set('threshold', 'manual');
model.result('pg3').feature('surf1').set('thresholdvalue', 0.2);
model.result('pg3').feature('surf1').set('colortable', 'Rainbow');
model.result('pg3').feature('surf1').set('colortabletrans', 'none');
model.result('pg3').feature('surf1').set('colorscalemode', 'linear');
model.result('pg3').feature('surf1').set('resolution', 'normal');
model.result('pg3').feature('surf1').set('colortable', 'Prism');
model.result('pg3').feature('surf1').create('def', 'Deform');
model.result('pg3').feature('surf1').feature('def').set('expr', {'u' 'v'});
model.result('pg3').feature('surf1').feature('def').set('descr', [native2unicode(hex2dec({'4f' '4d'}), 'unicode')  native2unicode(hex2dec({'79' 'fb'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode') ]);
model.result('pg1').run;
model.result('pg1').run;
model.result.export.create('data1', 'Data');
model.result.export('data1').setIndex('expr', 'acpr.p_t', 0);
model.result.export('data1').setIndex('unit', 'Pa', 0);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'60' '3b'}), 'unicode')  native2unicode(hex2dec({'58' 'f0'}), 'unicode')  native2unicode(hex2dec({'53' '8b'}), 'unicode') ], 0);
model.result.export('data1').setIndex('expr', 'solid.sGpxx', 1);
model.result.export('data1').setIndex('unit', 'N/m^2', 1);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'5e' '94'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'5f' '20'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'xx ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 1);
model.result.export('data1').setIndex('expr', 'solid.sGpxy', 2);
model.result.export('data1').setIndex('unit', 'N/m^2', 2);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'5e' '94'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'5f' '20'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'xy ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 2);
model.result.export('data1').setIndex('expr', 'solid.sGpyy', 3);
model.result.export('data1').setIndex('unit', 'N/m^2', 3);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'5e' '94'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode')  native2unicode(hex2dec({'5f' '20'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'yy ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 3);
model.result.export('data1').setIndex('expr', 'u', 4);
model.result.export('data1').setIndex('unit', 'mm', 4);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'4f' '4d'}), 'unicode')  native2unicode(hex2dec({'79' 'fb'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'X ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 4);
model.result.export('data1').setIndex('expr', 'v', 5);
model.result.export('data1').setIndex('unit', 'mm', 5);
model.result.export('data1').setIndex('descr', [native2unicode(hex2dec({'4f' '4d'}), 'unicode')  native2unicode(hex2dec({'79' 'fb'}), 'unicode')  native2unicode(hex2dec({'57' '3a'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode') 'Y ' native2unicode(hex2dec({'52' '06'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ], 5);
model.result.export('data1').setIndex('expr', 'rho_water', 6);
model.result.export('data1').setIndex('unit', '', 6);
model.result.export('data1').setIndex('descr', '', 6);

% model.result.export('data1').set('filename', 'D:\DATA\VA\Untitled.csv');  %path_result
path_result=['data\' num2str(parm_NN) '.csv'];  %path_result
model.result.export('data1').set('filename', path_result);

model.result.export('data1').set('location', 'regulargrid');
model.result.export('data1').set('regulargridx2', 128);
model.result.export('data1').set('regulargridy2', 128);
model.result.export('data1').run;

out = model;
