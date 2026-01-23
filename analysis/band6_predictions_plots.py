#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:00:06 2024

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import os
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.table import Table

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

direc = '/Users/jonty/mydata/robin/revised/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist if not (x.startswith('.'))]

filelist = mylistdir(direc+'../json_files/')

print(filelist)


#Generate values from pickle files

targets = []
q     = []
q_ep  = []
q_em  = []
md    = []
md_ep = []
md_em = []
ad    = []
ad_ep = []
ad_em = []
dstar = []
lstar = []
mstar = []

composition = 'temp2'

burnin = 1750

nfail = 0
for f in filelist:
    target = f.split('.')[0].strip()
        
    try:
        #print('Processing target ',target)

        sdb = open(direc+"../json_files/{:}.json".format(target))
        sdbjson = json.load(sdb)
        
        with open(direc+ 'output_'+composition+'/'+target+'_flatchain.json', 'r') as payload:
            sampler = json.load(payload)
        
        sampler = np.asarray(sampler)
        
        ndim = sampler.shape[1]

        if ndim == 4:
            labels=["mdust", "q", "amin", "epsilon"]
            for i in range(0,ndim-1): 
               

               
                if i == 0:
                    quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    md.append(quant[1])
                    md_ep.append(quant[2] - quant[1])
                    md_em.append(quant[1] - quant[0]) 

                if i == 1:
                    quant = np.quantile(sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    q.append(quant[1])
                    q_ep.append(quant[2] - quant[1])
                    q_em.append(quant[1] - quant[0])
                   
                if i == 2:
                    quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    ad.append(quant[1])
                    ad_ep.append(quant[2] - quant[1])
                    ad_em.append(quant[1] - quant[0])
            
        if ndim == 3:
            labels=["mdust", "amin", "epsilon"]
            for i in range(0,ndim-1): 
               
                quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
               
                if i == 0:
                    md.append(quant[1])
                    md_ep.append(quant[2] - quant[1])
                    md_em.append(quant[1] - quant[0]) 
                   
                if i == 1:
                    ad.append(quant[1])
                    ad_ep.append(quant[2] - quant[1])
                    ad_em.append(quant[1] - quant[0])
            
            q.append(-3.5)
            q_ep.append(0.0)
            q_em.append(0.0)

        #stellar parameters
        dstar.append(1./sdbjson["main_results"][0]["plx_arcsec"])
        lstar.append(sdbjson["main_results"][0]["lstar"])
        mstar.append(sdbjson["main_results"][0]["lstar"]**(1./3.5))
        targets.append(target)
        
    
    except:
        print('Target ',target,' sampler file not found.')
        nfail += 1
        
targets = np.asarray(targets)
dstar = np.asarray(dstar)
lstar = np.asarray(lstar)
mstar = np.asarray(mstar)
q     = abs(np.asarray(q))
q_ep  = np.asarray(q_ep)
q_em  = np.asarray(q_em)
md    = np.asarray(md)
md_ep = np.asarray(md_ep)
md_em = np.asarray(md_em)
ad    = np.asarray(ad)
ad_ep = np.asarray(ad_ep)
ad_em = np.asarray(ad_em)

rq = np.where(q != 3.5)

#Load Cao et al. 2023's data table - want names, ages

direc_cao = '/Users/jonty/mydata/cao2023/'
caotbl = ascii.read(direc_cao+'Table1.csv',delimiter=',')

#Remove footnotes
cao_name = caotbl['Names'].data
for i in range(len(cao_name)):
    cao_name[i] = cao_name[i].strip(' $^{grp}')

    cao_age = []
    
    for target in targets: #
        
        try: 
            cao_index = np.where(cao_name == target)
            
            cao_age.append(float(caotbl['Age'][cao_index][0]))
            #print(target, caotbl['Age'][cao_index][0])
            
        except:
            if target == 'GJ581':
                cao_age.append(2000)
            elif target == 'HD89452':
                cao_age.append(100)
            elif target == 'GJ3760':
                cao_age.append(1000)
            elif target == 'HD189002':
                cao_age.append(2035)
            elif target == 'TWA7':
                cao_age.append(10)
            elif target == 'HD216956C':
                cao_age.append(440)
            elif target == 'TYC93404371':
                cao_age.append(23)
            elif target == 'GJ14':
                cao_age.append(500)
            elif target == 'HD142315':
                cao_age.append(11)
            elif target == 'HD37594':
                cao_age.append(3)
            elif target == 'HD176894':
                cao_age.append(2120)
            elif target == 'HD89452':
                cao_age.append(100)
            elif target == 'HD202628':
                cao_age.append(2300)
            else :
                print('No age found for ',target,' adding fake')
                cao_age.append(-99)

cao_age = np.asarray(cao_age)

#REASONS target names
disc_extents = ascii.read(direc+'../targets/'+'resolved_discs_data_table_published_REASONS.csv')

reasons = disc_extents['Target']
origins = disc_extents['Origin']

#Read in flux predictions table
predictions = ascii.read(direc+'extrapolate/ALMA_predictions.tbl')

pred_names = predictions['target']
pred_fb6   = predictions['fb6']*1e3     #milliJanskies
pred_upfb6 = predictions['ufb6_hi']*1e3
pred_lofb6 = predictions['ufb6_lo']*1e3
pred_fb1   = predictions['fb1']*1e3     #microJanskies
pred_upfb1 = predictions['ufb1_hi']*1e3
pred_lofb1 = predictions['ufb1_lo']*1e3 

pred_age = []
pred_ls  = []
pred_ori = []
pred_rad = []
pred_wid = []
pred_inc = []
pred_pa  = []

for pname in pred_names:
    
    cao_index = np.where(targets == pname)
    
    pred_age.append(cao_age[cao_index])
    pred_ls.append(lstar[cao_index])

    reasons_index = np.where(reasons == pname)
    
    pred_ori.append(origins[reasons_index][0])
    pred_rad.append(disc_extents['R'][reasons_index][0])
    pred_wid.append(disc_extents['W'][reasons_index][0])
    pred_inc.append(disc_extents['pa'][reasons_index][0])
    pred_pa.append(disc_extents['inc'][reasons_index][0])


pred_age = np.asarray(pred_age)
pred_ls  = np.asarray(pred_ls)
pred_ori = np.asarray(pred_ori)
pred_rad = np.array(pred_rad)
pred_wid = np.array(pred_wid)
pred_inc = np.array(pred_inc)
pred_pa  = np.array(pred_pa)


pknown = np.where((pred_ori == 'ALMA/SMA'))
pposs  = np.where((pred_fb6+pred_upfb6 >= 300)&(pred_ori != 'ALMA/SMA'))
pcant  = np.where(pred_fb6+pred_upfb6 < 300)

import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')

c1 = cmap(0.25)
c2 = cmap(0.50)
c3 = cmap(0.85)

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.plot([0.01,100],[300,300],marker='',linestyle='--',color='black')
ax.errorbar(pred_ls[pknown],pred_fb6[pknown],xerr=None,yerr=[pred_lofb6[pknown],pred_upfb6[pknown]],marker='o',linestyle='',mec='white',mfc=c1,ecolor=c1,label='Observed')
ax.errorbar(pred_ls[pposs],pred_fb6[pposs],xerr=None,yerr=[pred_lofb6[pposs],pred_upfb6[pposs]],marker='o',linestyle='',mec='white',mfc=c2,ecolor=c2,label='Possible')
ax.errorbar(pred_ls[pcant],pred_fb6[pcant],xerr=None,yerr=[pred_lofb6[pcant],pred_upfb6[pcant]],marker='o',linestyle='',mec='white',mfc=c3,ecolor=c3,label='Faint')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.01,100.0])
ax.set_ylim([10,100000.0])
ax.set_xlabel(r'Stellar Luminosity ($L_{\odot}$)')
ax.set_ylabel(r'Flux Density ($\mu$Jy)')
ax.legend()
plt.tight_layout()
plt.savefig(direc+'band6_predictions_vs_lstar.pdf',dpi=200)
plt.show()
plt.close()

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.plot([1,10000],[300,300],marker='',linestyle='--',color='black')
ax.errorbar(pred_age[pknown],pred_fb6[pknown],xerr=None,yerr=[pred_lofb6[pknown],pred_upfb6[pknown]],marker='o',linestyle='',mec='white',mfc=c1,ecolor=c1,label='Observed')
ax.errorbar(pred_age[pposs],pred_fb6[pposs],xerr=None,yerr=[pred_lofb6[pposs],pred_upfb6[pposs]],marker='o',linestyle='',mec='white',mfc=c2,ecolor=c2,label='Possible')
ax.errorbar(pred_age[pcant],pred_fb6[pcant],xerr=None,yerr=[pred_lofb6[pcant],pred_upfb6[pcant]],marker='o',linestyle='',mec='white',mfc=c3,ecolor=c3,label='Faint')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1,10000.0])
ax.set_ylim([10,100000.0])
ax.set_xlabel(r'Stellar Age (Myr)')
ax.set_ylabel(r'Flux Density ($\mu$Jy)')
#ax.legend()
plt.tight_layout()
plt.savefig(direc+'band6_predictions_vs_age.pdf',dpi=200)
plt.show()
plt.close()

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.errorbar(pred_age[pknown],pred_ls[pknown],marker='o',linestyle='',mec='white',mfc=c1,ecolor=c1,label='Observed')
ax.errorbar(pred_age[pposs],pred_ls[pposs],marker='o',linestyle='',mec='white',mfc=c2,ecolor=c2,label='Possible')
ax.errorbar(pred_age[pcant],pred_ls[pcant],marker='o',linestyle='',mec='white',mfc=c3,ecolor=c3,label='Faint')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1,10000.0])
ax.set_ylim([0.01,100.0])
ax.set_xlabel(r'Stellar Age (Myr)')
ax.set_ylabel(r'Stellar Luminosity ($L_{\odot}$)')
#ax.legend()
plt.tight_layout()
plt.savefig(direc+'band6_lstar_vs_age.pdf',dpi=200)
plt.show()
plt.close()


#Output tables
b6poss_names = pred_names[pposs]
b6poss_flux  = pred_fb6[pposs]
b6poss_errp  = pred_upfb6[pposs]
b6poss_errm  = pred_lofb6[pposs]
b6poss_rad = pred_rad[pposs]
b6poss_wid = pred_wid[pposs]
b6poss_inc = pred_inc[pposs]
b6poss_pa  = pred_pa[pposs]
b6poss_ra  = []
b6poss_dec = []
b6poss_dist = []
b6poss_udis = []

simbad = Simbad()
simbad.add_votable_fields("mesplx")

for i in range(0,len(b6poss_names)):
    if b6poss_names[i] == 'HD216956C' : 
        result_table = simbad.query_object('Fomalhaut C')
    else: 
        result_table = simbad.query_object(b6poss_names[i])
    
    b6poss_ra.append(result_table['ra'].value[0])
    b6poss_dec.append(result_table['dec'].value[0])
    
    dist  = 1000./result_table['mesplx.plx'].value[0]
    udist = 1000./(result_table['mesplx.plx'].value[0] - result_table['mesplx.plx_err'].value[0]) - dist
    
    b6poss_dist.append(dist)
    b6poss_udis.append(udist)

b6poss_ra   = np.asarray(b6poss_ra)
b6poss_dec  = np.asarray(b6poss_dec)
b6poss_dist = np.asarray(b6poss_dist)
b6poss_udis = np.asarray(b6poss_udis)

b6poss_table = Table([b6poss_names,b6poss_dist,b6poss_udis,b6poss_ra,b6poss_dec,b6poss_flux,b6poss_errp,b6poss_errm,b6poss_rad,b6poss_wid,b6poss_inc,b6poss_pa],\
                     names=('Target','Distance (pc)','Distance error (pc)','RA (deg)','DEC (deg)','Flux B6 (uJy)','upFlux B6 (uJy)','udFlux B6 (uJy)','Radius (au)','Width (au)','inc (deg)','pa (deg)'))

b6poss_table.write(direc + 'B6_potentially_observable_sources.tbl',format='ascii.ecsv',overwrite=True)

pposs  = np.where((pred_fb1+pred_upfb1 >= 15))

b1poss_names = pred_names[pposs]
b1poss_flux  = pred_fb1[pposs]
b1poss_errp  = pred_upfb1[pposs]
b1poss_errm  = pred_lofb1[pposs]
b1poss_rad = pred_rad[pposs]
b1poss_wid = pred_wid[pposs]
b1poss_inc = pred_inc[pposs]
b1poss_pa  = pred_pa[pposs]
b1poss_ra  = []
b1poss_dec = []
b1poss_dist = []
b1poss_udis = []

simbad = Simbad()
simbad.add_votable_fields("mesplx")

for i in range(0,len(b1poss_names)):
    
    if b1poss_names[i] == 'HD216956C' : 
        result_table = simbad.query_object('Fomalhaut C')
    else: 
        result_table = simbad.query_object(b1poss_names[i])
    
    b1poss_ra.append(result_table['ra'].value[0])
    b1poss_dec.append(result_table['dec'].value[0])
    
    dist  = 1000./result_table['mesplx.plx'].value[0]
    udist = 1000./(result_table['mesplx.plx'].value[0] - result_table['mesplx.plx_err'].value[0]) - dist
    
    b1poss_dist.append(dist)
    b1poss_udis.append(udist)

b1poss_ra   = np.asarray(b1poss_ra)
b1poss_dec  = np.asarray(b1poss_dec)
b1poss_dist = np.asarray(b1poss_dist)
b1poss_udis = np.asarray(b1poss_udis)

b1poss_table = Table([b1poss_names,b1poss_dist,b1poss_udis,b1poss_ra,b1poss_dec,b1poss_flux,b1poss_errp,b1poss_errm,b1poss_rad,b1poss_wid,b1poss_inc,b1poss_pa],\
                     names=('Target','Distance (pc)','Distance error (pc)','RA (deg)','DEC (deg)','Flux B1 (uJy)','upFlux B1 (uJy)','udFlux B1 (uJy)','Radius (au)','Width (au)','inc (deg)','pa (deg)'))

b1poss_table.write(direc + 'B1_potentially_observable_sources.tbl',format='ascii.ecsv',overwrite=True)
