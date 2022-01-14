"""
Functions for feature importance calculations.

- Pearson, Spearman, Kendall
- (Power scaled) Bayesian Linear Regression
- Permutation based: Random Forest Permutation
- model-agnostic correaltion coefficients

Copyright 2022 Sebastian Haan, The University of Sydney

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import itertools
import sys
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from xicor.xicor import Xi
#import json
#from tqdm import tqdm



def plot_feature_correlation_spearman(X, feature_names, outpath, show = FALSE):
	"""
	Plot feature correlations using Spearman correlation coefficients.
	Feature correlations are automatically clustered using hierarchical clustering.

	Result figure is automatically saved in specified path.

	Input:
		X: data array
		feature names: list of feature names
		outpath: path to save plot
		show: if True, interactive matplotlib plot is shown
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	corr = spearmanr(X).correlation
	corr_linkage = hierarchy.ward(corr)
	dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90)
	dendro_idx = np.arange(0, len(dendro['ivl']))

	# Plot results:
	pos = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
	ax2.set_xticks(dendro_idx)
	ax2.set_yticks(dendro_idx)
	ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
	ax2.set_yticklabels(dendro['ivl'])
	fig.colorbar(pos, ax = ax2)
	fig.tight_layout()
	plt.savefig(os.path.join(outpath, 'Feature_Correlations_Hierarchical_Spearman.png'), dpi = 300)
	if show:
		plt.show()



def calc_new_correlation(X, y):
	"""
	Calculation of general correlation coefficient 
	as described in Chatterjee, S. (2019, September 22):
	A new coefficient of correlation. arxiv.org/abs/1909.10140
	Returns correlation coeeficient for testing independence

	In comparison to Spearman, Pearson, and Kendalls, this new correlation coeffcient
	can measure associations that are not monotonic or non-linear, 
	and is 0 if X and Y are independent, and is 1 if there is a function Y= f(X).
	Note that this coeffcient is intentionally not summetric, because we want to understand 
	if Y is a function X, and not just if one of the variables is a function of the other.
	
	This function also removes Nans and converts factor variables to integers automatically.

	Input:
		X: data feature array with shape (npoints, n_features)
		y: target variable as vector with size npoints

	Return:
		corr: correlation coefficients
	"""
	n_features = X.shape[1] 
	corr = np.empty(n_features)
	pvals = np.empty(n_features)
	for i in range(n_features):
		xi_obj = Xi(X[:,i], y)
		corr[i] = xi_obj.correlation
		pvals[i] = xi_obj.pval_asymptotic(ties=False, nperm=1000)
	# set correlation coefficient to zero for non-significant p_values (P > 0.01)
	corr[pvals>0.01] = 0
	return corr


def test_calc_new_correlation():
	"""
	Test function for calc_new_correlation
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.01)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	corr = calc_new_correlation(X, y)
	assert np.argmax(coefsim) == np.argmax(corr)


def blr_factor_importance(X_train, y_train, logspace = False, signif_threshold = 2):
	"""
	Trains Bayesian Linear Regresssion model and returns the estimated significance of regression coeffcients.
	The significance of the linear coefficient is defined by dividing the estimated coefficient 
	over the standard deviation of this estimate. The correlation significance is set to zero if below threshold.

	Input:
		X: input data matrix with shape (npoints,nfeatures)
		y: target varable with shape (npoints)
		logspace: if True, models regression in logspace
		signif_threshold: threshold for coefficient significance to be considered significant (Default = 2)
		

	Return:
		coef_signif: Signigicance of coefficients (Correlation coeffcient / Stddev)
	"""
	if logspace:
		x = np.log(X_train)
		y = np.log(y_train)
	else:
		x = X_train
		y = y_train
	#sel = np.where(np.isfinite(x) & np.isfinite(y))
	if x.shape[1] == 1:
		x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	reg = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True)
	reg.fit(x, y)

	#print('BLR regresssion coefficients:')
	# Set not significant coeffcients to zero
	coef = reg.coef_.copy()
	coef_sigma = np.diag(reg.sigma_).copy()
	coef_signif = coef / coef_sigma
	#for i in range(len(coef)):
	#	print('X' + str(i), ' wcorr=' + str(np.round(coef[i], 3)) + ' +/- ' + str(np.round(coef_sigma[i], 3)))
	# Set not significant coeffcients to zero:
	coef_signif[coef_signif < signif_threshold] = 0
	return coef_signif


def test_blr_factor_importance():
	"""
	Test function for blr_factor_importance
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'linear', noise = 0.05)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	coef_signif = blr_factor_importance(X, y)
	assert np.argmax(coefsim) == np.argmax(coef_signif)


def rf_factor_importance(X_train, y_train, correlated = False):
	"""
	Factor importance using RF permutation test and optional corrections 
	for multicollinarity (correlated) features. 
	Including training of Random Forest regression model with trainig data 
	and setting non-significant coefficients to zero.

	Input:
		X: input data matrix with shape (npoints,nfeatures)
		y: target varable with shape (npoints)
		correlated: if True, features are assumed to be correlated

	Return:
		imp_mean_corr: feature importances
	"""
	rf_reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state = 42)
	rf_reg.fit(X_train, y_train)
	result = permutation_importance(rf_reg, X_train, y_train, n_repeats=20, random_state=42, 
		n_jobs=1, scoring = "neg_mean_squared_error")
	imp_mean = result.importances_mean
	imp_std = result.importances_std
	# Make corrections for correlated features
	# This is neccessary since permutation importance are lower for correlated features
	if correlated:
		corr = spearmanr(X_train).correlation
		imp_mean_corr = np.zeros(len(imp_mean))
		imp_std_corr = np.zeros(len(imp_mean))
		for i in range(len(imp_mean)):
			imp_mean_corr[i] = np.sum(abs(corr[i]) * imp_mean)
			imp_std_corr[i] = np.sqrt(np.sum(abs(corr[i]) * imp_std**2))
	else:
		imp_mean_corr = imp_mean
		imp_std_corr = imp_std
	#print("Random Forest factor importances: ", imp_mean_corr)
	#print("Random Forest factor importances std: ", imp_std_corr)
	# Set non significant features to zero:
	imp_mean_corr[imp_mean_corr / imp_std_corr < 3] = 0
	imp_mean_corr[imp_mean_corr < 0.001] = 0
	return imp_mean_corr


def test_rf_factor_importance():
	"""
	Test function for rf_factor_importance
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.05)
	X = dfsim[feature_names].values
	y = dfsim['Ytarget'].values
	imp_mean_corr = rf_factor_importance(X, y)
	assert np.argmax(coefsim) == np.argmax(imp_mean_corr)



def create_simulated_features(n_features, outpath = None, n_samples = 200, model_order = 'quadratic', correlated = False, noise= 0.1):
	"""
	Generate synthetic datasets for testing

	see also https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

	Input:
		n_features: number of features
		outpath: path to save simulated data
		n_samples: number of samples	
		model_order: order of the model, either 'linear', 'quadratic', or 'cubic'
		correlated: if True, the features are correlated
		noise: noise level [range: 0-1]

	Return:
		dfsim: dataframe with simulated features
		coefsim: simulated coefficients
		feature_names: list of feature names
	"""
	if correlated:
		n_rank = int(n_features/2)
	else:
		n_rank = None
	Xsim, ysim, coefsim = make_regression(n_samples=200, n_features = n_features, n_informative=int(n_features/2), n_targets=1, 
		bias=0.5, noise=noise, shuffle=False, coef=True, random_state=42, effective_rank = n_rank)	
	feature_names = ["Feature_" + str(i+1) for i in range(n_features)]
	coefsim /= 100
	scaler = MinMaxScaler()
	scaler.fit(Xsim)
	Xsim = scaler.transform(Xsim)
	if outpath is not None:
		plot_feature_correlation(Xsim, feature_names, outpath)
		# PLot all correlations
		sorted_idx = coefsim.argsort()
		fig, ax = plt.subplots(figsize = (6,5))
		ypos = np.arange(len(coefsim))
		bar = ax.barh(ypos, coefsim[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
		gradientbars(bar, coefsim[sorted_idx])
		plt.xlabel("True Feature Coefficients")
		plt.tight_layout()
		plt.savefig(os.path.join(outpath, 'Feature_True_Coef.png'), dpi = 300)
		plt.close('all')
	#plot_feature_correlation(Xsim, feature_names)
	# make first-order model
	if model_order == 'linear':
		ysim_new = np.dot(Xsim, coefsim) + np.random.normal(scale=noise, size = n_samples)
	elif model_order == 'quadratic':
		# make quadratic model
		Xcomb = []
		for i, j in itertools.combinations(Xsim.T, 2):
			Xcomb.append(i * j) 
		Xcomb = np.asarray(Xcomb).T
		Xcomb = np.hstack((Xsim, Xcomb, Xsim**2))
		coefcomb = []
		for i, j in itertools.combinations(coefsim, 2):
			coefcomb.append(i * j) 
		coefcomb = np.asarray(coefcomb)
		coefcomb = np.hstack((coefsim, coefcomb, coefsim**2))
		ysim_new = np.dot(Xcomb, coefcomb) + np.random.normal(scale=noise, size = n_samples)
	elif model_order == 'quadratic_pairwise':
		# make quadratic model
		Xcomb = []
		for i, j in itertools.combinations(Xsim.T, 2):
			Xcomb.append(i * j) 
		Xcomb = np.asarray(Xcomb).T
		Xcomb = np.hstack((Xsim, Xcomb))
		coefcomb = []
		for i, j in itertools.combinations(coefsim, 2):
			coefcomb.append(i * j) 
		coefcomb = np.asarray(coefcomb)
		coefcomb = np.hstack((coefsim, coefcomb))
		ysim_new = np.dot(Xcomb, coefcomb) + np.random.normal(scale=noise, size = n_samples)
	#Save data as dataframe and coefficients on file
	header = np.hstack((feature_names, 'Ytarget'))
	data = np.hstack((Xsim, ysim_new.reshape(-1,1)))
	df = pd.DataFrame(data, columns = header)
	if outpath is not None:
		os.makedirs(outpath, exist_ok=True)
		df.to_csv(os.path.join(outpath, f'SyntheticData_{model_order}_{n_features}nfeatures_{noise}noise'))
		df_coef = pd.DataFrame(coefsim.reshape(-1,1).T, columns = feature_names)
		df_coef.to_csv(os.path.join(outpath, f'SyntheticData_coefficients_{model_order}_{n_features}nfeatures_{noise}noise'), index = False)
	return df, coefsim, feature_names


def gradientbars(bars, data):
	"""
	Helper function for making colorfull bars

	Input:
		bars: list of bars
		data: data to be plotted
	"""
	ax = bars[0].axes
	lim = ax.get_xlim()+ax.get_ylim()
	for bar in bars:
		bar.set_zorder(1)
		bar.set_facecolor("none")
		x,y = bar.get_xy()
		w, h = bar.get_width(), bar.get_height()
		grad = np.atleast_2d(np.linspace(0,1*w/max(data),256))
		ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, norm=mpl.colors.NoNorm(vmin=0,vmax=1))
	ax.axis(lim)


def plot_correlationbar(corrcoefs, feature_names, outpath, fname_out, name_method = None, show = False):
	"""
	Helper function for plotting feature correlation.
	Result plot is saved in specified directory.

	Input:
		corrcoefs: list of feature correlations
		feature_names: list of feature names
		outpath: path to save plot
		fname_out: name of output file (should end with .png)
		name_method: name of method used to compute correlations
		show: if True, show plot
	"""
	sorted_idx = corrcoefs.argsort()
	fig, ax = plt.subplots(figsize = (6,5))
	ypos = np.arange(len(corrcoefs))
	bar = ax.barh(ypos, corrcoefs[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
	gradientbars(bar, corrcoefs[sorted_idx])
	if name_method is not None:	
		plt.title(f'Feature Correlations for {name_method}')	
	plt.xlabel("Feature Importance")
	plt.tight_layout()
	plt.savefig(os.path.join(outpath, fname_out), dpi = 300)
	if show:
		plt.show()
	plt.close('all')


def test_plot_correlationbar(outpath):
	"""
	Test function for plot_correlationbar
	"""
	dfsim, coefsim, feature_names = create_simulated_features(6, model_order = 'quadratic', noise = 0.05)
	plot_correlationbar(coefsim, feature_names, outpath, 'test_plot_correlationbar.png', show = True)


"""
#def main(): 

# load settings:
from config_loader import * 

outpath = os.path.join(outpath, 'results_feature_importance')

if use_simdata:
	os.makedirs(outpath, exist_ok = True)
	dftrain, coefsim, feature_names = create_simulated_features(10, outpath, n_samples = 200, model_order = sim_model, 
		noise = sim_noise, correlated = True)
	name_features2 = feature_names
	name_target = 'Ytarget'

else:
	os.makedirs(outpath, exist_ok = True)
	# Pre-process data
	print('Reading and pre-processing data...')
	dfsel, name_features2 = preprocess(inpath, infname, outfname, name_target, name_features, zmin = 100*zmin, zmax= 100*zmax, categorical = 'Soiltype',
	colname_depthmin = colname_depthmin, colname_depthmax = colname_depthmax)
	#dfsel = pd.read_csv(os.path.join(inpath,outfname))
	#name_features2.extend(['z'])
	# split into train and test data
	dftrain = dfsel.copy()
	y_train = dftrain[name_target].values

print("Calculate feature correlation plot...")
plot_feature_correlation(dftrain[name_features2+[name_target]].values, name_features2+[name_target], outpath)


print("----Calculating Feature Importance----")
print("")
print("Hold down your bagel.")
print("")
print('Calculating factor importance from BLR..')
X_train = dftrain[name_features2].values
y_train = dftrain[name_target].values
# Scale data with standardscaler:
Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train, scaler = 'standard')
# Train BLR and get coeefcients
corrcoef_blr, corrcoef_std_blr = blr.blr_factor_importance(Xs_train, ys_train)
# Plot Results
sigma_blr = abs(corrcoef_blr) / corrcoef_std_blr
sorted_idx = sigma_blr.argsort()
fig, ax = plt.subplots(figsize = (6,5))
ypos = np.arange(len(sigma_blr))
bar = ax.barh(ypos, sigma_blr[sorted_idx], tick_label = np.asarray(name_features2)[sorted_idx], align='center')
gradientbars(bar, sigma_blr[sorted_idx])
plt.xlabel("Linear BLR Feature Significance [sigmas]")
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'Feature_Significance_linearBLR.png'), dpi = 300)
# Scale data with powerscaler
Xs_train, ys_train, scale_params = blr.scale_data(X_train, y_train, scaler = 'power')
# Train BLR and get coeefcients
corrcoef_blr, corrcoef_std_blr = blr.blr_factor_importance(Xs_train, ys_train)
# Plot Results
sigma_blr = abs(corrcoef_blr) / corrcoef_std_blr
sorted_idx_blr = sigma_blr.argsort()
fig, ax = plt.subplots(figsize = (6,5))
ypos = np.arange(len(sigma_blr))
bar = ax.barh(ypos, sigma_blr[sorted_idx_blr], tick_label = np.asarray(name_features2)[sorted_idx_blr], align='center')
gradientbars(bar, sigma_blr[sorted_idx_blr])
plt.xlabel("Power-law BLR Feature Significance [sigmas]")
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'Feature_Significance_powerBLR.png'), dpi = 300)



print('Calculating factor importance from Random Forest Regressor based on permutation...')
X_train = dftrain[name_features2].values
y_train = dftrain[name_target].values
corrcoef_rf,corrcoef_rf_std = rf.rf_factor_importance2(X_train, y_train)
corrcoef_rf = np.asarray(corrcoef_rf)
corrcoef_rf_std = np.asarray(corrcoef_rf_std)
# Plot Results
sorted_idx_rf = corrcoef_rf.argsort()
fig, ax = plt.subplots(figsize = (6,5))
ypos = np.arange(len(corrcoef_rf))
bar = ax.barh(ypos, corrcoef_rf[sorted_idx_rf], xerr = corrcoef_rf_std[sorted_idx_rf],
tick_label = np.asarray(name_features2)[sorted_idx_rf], align='center')
gradientbars(bar, corrcoef_rf[sorted_idx_rf])
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'Feature_Importance_RF_permutation.png'), dpi = 300)
plt.close('all')



# Visualise main results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (6,9))
bar1 = ax1.barh(np.arange(len(corrcoef_rf)), corrcoef_rf[sorted_idx_rf], xerr = corrcoef_rf_std[sorted_idx_rf],
	tick_label = np.asarray(name_features2)[sorted_idx_rf], align='center')
gradientbars(bar1, corrcoef_rf[sorted_idx_rf])
ax1.set_xlabel("Random Forest Feature Importance")
bar2 = ax2.barh(np.arange(len(sigma_blr)), sigma_blr[sorted_idx_blr], tick_label = np.asarray(name_features2)[sorted_idx_blr], align='center')
gradientbars(bar2, sigma_blr[sorted_idx_blr])
ax2.set_xlabel("Bayesian Regression Feature Significance [sigmas]")
plt.tight_layout()
plt.show()
#plt.show(block=False)
#plt.pause(10)
#plt.close('all')



#if __name__ == '__main__':
#	# execute main script:
#	main()

"""