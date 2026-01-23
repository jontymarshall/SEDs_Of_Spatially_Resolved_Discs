#!/usr/bin/env python3
#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:14:30 2018

@author: jonty
"""
import os
import emcee
import corner
import dill
import json
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import miepython.miepython as mpy
from pathos.multiprocessing import Pool
from numba import jit
import time
from astropy.io import ascii
from scipy import interpolate
from scipy import integrate

from RT_Code import RTModel

os.environ["OMP.NUM.THREADS"] = "75"

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

direc = '/home/jmarshall/sed_fitting/'


#constants
h = 6.626e-34
c = 299792458.0 # m/s
k = 1.38e-23
sb = 5.67e-8 # 
au     = 1.495978707e11 # m 
pc     = 3.0857e16 # m
lsol   = 3.828e26 # W
rsol   = 6.96342e8 # m
MEarth = 5.97237e24 # kg

um = 1e-6 #for wavelengths in microns

def model(theta, x, y, yerr,radius, width):

    if len(theta) == 4:
        mdust, q, amin, epsilon = theta
    if len(theta) == 3:
        mdust, amin, epsilon = theta
        q = -3.5   

    model = RTModel()
    
    RTModel.get_parameters(model,'RTModel_Input_File.txt')

    model.parameters['directory'] = '/home/jmarshall/sed_fitting/output/'
    model.parameters['prefix'] = 'sed_fitting_'
    model.parameters['stype'] = 'json'
    model.parameters['model'] = direc + 'json/' + inputname
    model.parameters['tstar'] = sdbjson['main_results'][0]['Teff']
    model.parameters['rstar'] = sdbjson['main_results'][0]['rstar']
    model.parameters['dstar'] = (1/sdbjson['main_results'][0]['plx_arcsec'])
    model.parameters['lstar'] = sdbjson['main_results'][0]['lstar']
    
    model.obs_wave = x
    model.obs_flux = y  # [x*1000 for x in y]
    model.obs_uncs = yerr # [x*1000 for x in yerr]

    #Dust
    model.parameters['dtype'] = 'gauss'
    model.parameters['mdust'] = 10**mdust
    model.parameters['rpeak'] = radius
    model.parameters['rfwhm'] = width
    if len(theta) == 4:
        model.parameters['q'] = q
    if len(theta) == 3:
        model.parameters['q'] = -3.5
    model.parameters['amin']  = 10**amin
    model.parameters['alpha_out'] = 0
    
    RTModel.make_star(model)
    #RTModel.scale_star(model)
    RTModel.make_dust(model)
    RTModel.make_disc(model)
    RTModel.calculate_surface_density(model)
    RTModel.read_optical_constants(model)
    RTModel.calculate_qabs(model)
    RTModel.calculate_dust_emission(model,mode='full',tolerance=1e-3)
    RTModel.calculate_dust_scatter(model)
    RTModel.flam_to_fnu(model)
    
    mdl = model.sed_wave,model.sed_star + model.sed_emit + model.sed_scat
    model_specs.append(mdl)

    return model

def lnlike(theta, x, y, yerr,radius,width):
    
    if len(theta) == 4:
        model_lnlike = model(theta, x, y, yerr,radius,width)
        dum1,dum2,dum3,epsilon = theta
    if len(theta) == 3: 
        model_lnlike = model(theta, x, y, yerr,radius,width)
        dum1,dum2,epsilon = theta

    f = interpolate.interp1d(model_lnlike.sed_wave, model_lnlike.sed_star + model_lnlike.sed_emit + model_lnlike.sed_scat, fill_value="extrapolate")
    #y = np.array([x*1000 for x in y])
    #yerr = np.array([x*1000 for x in yerr])
    #yerr_rel = y/yerr
    longwl = np.where(x >= 10.0) #cut at lambda < 10 um

    sigma2 = yerr[longwl]**2 + f(x[longwl])**2 * np.exp(2 * epsilon)
    lnlike = -0.5 * np.sum((y[longwl] - f(x[longwl]))**2 /sigma2 + np.log(sigma2))
    return lnlike

def lnprior(theta):
    if len(theta) == 4:
        mdust, q, amin, epsilon = theta
    if len(theta) == 3:
        mdust, amin, epsilon = theta
        q = -3.5
    
    if (1e-6 < 10**mdust < 1e1 and -4 < q < -3 and 0.05 < 10**amin < 50.0 and -10 < epsilon < 1):
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, yerr,radius,width):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr,radius,width) 

def run_emcee(sampler,pos,ndim,burn,steps):
    print("\nRunning burn in")
    p0, prob, state = sampler.run_mcmc(pos, burn, progress=True)
    sampler.reset()
    print("\nRunning production")
    sampler.run_mcmc(p0, steps, rstate0=np.random.get_state(), progress=True)
    print("Done.")
    return sampler

#Plotting for results
def plotter(sampler,x,y,yerr):
        #y = np.array([x*1000 for x in y]) 
        #yerr = np.array([x*1000 for x in yerr]) 

        plt.plot(x,y, zorder=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength in log microns')
        plt.ylabel('Flux in log mJy')
        plt.title('{:} MCMC'.format(inputname.split('.')[0]))
        plt.errorbar(x, y, xerr=None, yerr=yerr,marker='.',
                  linestyle='',mec='black', mfc='white', fmt='none',
                  label="Photometry Uncertainty", zorder=10)
        for spec in model_specs:
            plt.plot(spec[0], spec[1], color="r", alpha=0.05)
        plt.ylim(1e-1,np.max(spec[1])*2.5)
        plt.savefig(direc+'output/{:}_plot.png'.format(inputname.split('.')[0]))
        plt.close()

def stepplot(sampler):

        if ndim == 4:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples = sampler.get_chain()
            tauto = sampler.get_autocorr_time(quiet=True)
            labels = ["mdust", "q", "amin","epsilon"]
            axes[0].set_yscale('log')
            
            for i in range(ndim):
                xlims = [0, len(samples)]
                ylims = [np.min(samples[:, :, i]),np.max(samples[:, :, i])]
                for x in range(nwalkers):
                    if samples[:, :, 0][:, x][0] != samples[:, :, 0][:, x][-1]:
                        ax = axes[i]
                        ax.plot(samples[:, :, i][:, x], "k", alpha=0.3)
                        ax.set_xlim(0, len(samples))
                        ax.set_ylabel(labels[i])
                    else:
                        ax = axes[i]
                        ax.plot(samples[:, :, i][:, x], "darkred", alpha=0.3)
                        ax.set_xlim(0, len(samples))
                        ax.set_ylabel(labels[i])

                ax.plot([10*tauto[i],10*tauto[i]],[ylims[0],ylims[1]],color="red",marker="",linestyle="-")

        if ndim == 3:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples = sampler.get_chain()
            tauto = sampler.get_autocorr_time(quiet=True)
            labels = ["mdust", "amin","epsilon"]
            axes[0].set_yscale('log')
            
            for i in range(ndim):
                xlims = [0, len(samples)]
                ylims = [np.min(samples[:, :, i]),np.max(samples[:, :, i])]
                for x in range(nwalkers):
                    if samples[:, :, 0][:, x][0] != samples[:, :, 0][:, x][-1]:
                        ax = axes[i]
                        ax.plot(samples[:, :, i][:, x], "k", alpha=0.3)
                        ax.set_xlim(0, len(samples))
                        ax.set_ylabel(labels[i])
                    else:
                        ax = axes[i]
                        ax.plot(samples[:, :, i][:, x], "darkred", alpha=0.3)
                        ax.set_xlim(0, len(samples))
                        ax.set_ylabel(labels[i])

                ax.plot([10*tauto[i],10*tauto[i]],[ylims[0],ylims[1]],color="red",marker="",linestyle="-")

        axes[-1].set_xlabel("Step number")
        plt.suptitle('{:} Walkers Step Plot'.format(inputname.split('.')[0]))
        plt.savefig(direc+'output/{:}_stepplot'.format(inputname.split('.')[0]))
        plt.close()

def cornerplot(sampler):
        flat_samples = sampler.flatchain
        if ndim == 4:
            fig = corner.corner(flat_samples, labels=["mdust", "q", "amin", "epsilon"], quantiles=[0.16, 0.5, 0.84],show_titles=True)
        if ndim == 3:
            fig = corner.corner(flat_samples, labels=["mdust", "amin", "epsilon"], quantiles=[0.16, 0.5, 0.84],show_titles=True)

        plt.suptitle('{:} Corner Plot'.format(inputname.split('.')[0]))
        plt.savefig(direc+'output/{:}_cornerplot'.format(inputname.split('.')[0]))
        plt.close()

def bestfit(sampler, x,y,yerr):
        #y = [x*1000 for x in y]
        #yerr = [x*1000 for x in yerr]

        samples = sampler.flatchain
        theta_max = samples[np.argmax(sampler.flatlnprobability)]
        best_fit_model = model(theta_max,x,y,yerr,radius,width)

        plt.plot(x, y, 'ro', markersize=3, label='Photometry', zorder=10, alpha=0.5)
        plt.errorbar(x, y, xerr=None, yerr=yerr,marker='.',
                  linestyle='',mec='black', mfc='white', fmt='none', zorder=15)
        plt.plot(best_fit_model.sed_wave,best_fit_model.sed_star + best_fit_model.sed_emit + best_fit_model.sed_scat,
                 label='Highest Likelihood Model', color='turquoise', zorder=5, alpha=0.7)
        plt.plot(best_fit_model.sed_wave,best_fit_model.sed_emit, 
                 label='Fitted Cold Disk', color='steelblue', linestyle='dashed')
        plt.plot(best_fit_model.sed_wave,best_fit_model.sed_star, 
                 label='Star', color='goldenrod', linestyle='dashed')
        plt.ylim(1e-1,np.max(np.sum(best_fit_model.sed_star + best_fit_model.sed_emit + best_fit_model.sed_scat))*2.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Wavelength in log microns')
        plt.ylabel('Flux in log mJy')
        plt.title('{:} Best Fit Plot'.format(inputname.split('.')[0]))
        plt.savefig(direc+'output/{:}_bestfitplot'.format(inputname.split('.')[0]), dpi=200)
        plt.close()

#Start doing actual emcee things

#list of targets
filelist = os.listdir(direc+'json/')
print(filelist)

#data on each target disc extent
disc_extents = ascii.read(direc+'resolved_discs_data_table.csv')

targets = disc_extents['Target'].data
radii   = disc_extents['R'].data
wides   = disc_extents['W'].data

icount = 0
for f in filelist:

        inputname = f
        file = open(direc+"json/{:}".format(inputname))
        sdbjson = json.load(file)

        print("Processing target "+inputname+" which is "+str(int(icount+1))+" of "+str(len(filelist))+" targets.")

        #stellar parameters
        dstar = 1./sdbjson["main_results"][0]["plx_arcsec"]
        lstar = sdbjson["main_results"][0]["lstar"] 

        #stellar photosphere
        obs_photosphere = np.array(sdbjson['star_spec']['fnujy'])
        obs_lambda = np.array(sdbjson['star_spec']['wavelength'])

        #omit upper limits and duplicates from observations to be fitted
        ignore = np.asarray(sdbjson['phot_ignore'][0])
        upperlimit = np.asarray(sdbjson['phot_upperlim'][0])

        good = np.where((ignore == False)&(upperlimit == False))

        x = np.asarray(sdbjson['phot_wavelength'][0])[good]
        y = np.asarray(sdbjson['phot_fnujy'][0])[good]*1e3 #modelling in mJy, inputs in Jy
        yerr = np.asarray(sdbjson['phot_e_fnujy'][0])[good]*1e3 #modelling in mJy, inputs in Jy

        #Disc properties
        index = np.where(targets == inputname.split('.')[0])
        target = targets[index]
        radius = radii[index]
        width  = wides[index]

        #Remove inner warm component from disc - replace with proper filters!
        try:
                xwarm = np.array(sdbjson["model_spectra"][2]["wavelength"])
                ywarm = np.array(sdbjson["model_spectra"][2]["fnujy"])

                #print(xwarm,ywarm)

                f = interpolate.interp1d(xwarm,ywarm*1e3)
                yhot = f(x)

                y -= yhot
        except:
                print("No warm component for target ",inputname)


        manager = mp.Manager()
        model_specs = manager.list()
            
        nwalkers = 40
        nburn = 100
        nsteps = 1000
        initial = np.array([-3, -3.5, 0.0, 0.0]) #implement log space?
        ndim = len(initial)
        p0 = [np.array(initial) for i in range(nwalkers)]

        #if enough sub-mm data exists, try to find q
        if np.max(x) >= 350:
                for aa in range(nwalkers):

                        p0[aa] = np.array([np.random.uniform(-6, -2),
                                           np.random.uniform(-3.9, -3.1),
                                           np.random.uniform(-1, 1),
                                           np.random.uniform(-10,1)])

                data = (x, y, yerr,radius,width)

                start = time.time()
                print('Running emcee sampler for {:}'.format(inputname))

                with mp.Pool() as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
                    results = run_emcee(sampler,p0,ndim,nburn,nsteps)
                        
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                        
                dill.dump(sampler, open(direc+'output/{:}_sampler.pickle'.format(inputname.split('.')[0]),"wb"))
                with open(direc+'output/{:}_flatchain.json'.format(inputname.split('.')[0]), 'w') as fp:
                    json.dump(sampler.flatchain.tolist(), fp, indent=4)
                dill.dump(model_specs, open('output/{:}_sed.pickle'.format(inputname.split('.')[0]),"wb"))
            
        #if no sub-mm exists, then omit q from fitting, and assume q = -3.5
        else:
                initial = np.array([-3, 0.0, 0.0])
                ndim = len(initial)
                p0 = [np.array(initial) for i in range(nwalkers)]

                for aa in range(nwalkers):
                    p0[aa] = np.array([np.random.uniform(-6, -2),
                                       np.random.uniform(-1,1),
                                       np.random.uniform(-10,1)])
                data = (x, y, yerr, radius, width)
                
                print("Lacking long wavelength observations")
                
                start = time.time()
                print('Running emcee sampler for {:}'.format(inputname.split('.')[0]))
                
                with mp.Pool() as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
                    results = run_emcee(sampler,p0,ndim,nburn,nsteps)
                
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                
                dill.dump(sampler, open(direc+'output/{:}_sampler.pickle'.format(inputname.split('.')[0]),"wb"))
                with open(direc+'output/{:}_flatchain.json'.format(inputname.split('.')[0]), 'w') as fp:
                    json.dump(sampler.flatchain.tolist(), fp, indent=4)
                dill.dump(model_specs, open('output/{:}_sed.pickle'.format(inputname.split('.')[0]),"wb"))
        
        plotter(sampler,x,y,yerr)
        stepplot(sampler)
        bestfit(sampler,x,y,yerr)
        cornerplot(sampler)
        icount += 1
