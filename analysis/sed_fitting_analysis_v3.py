import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import os
import corner
from astropy.io import ascii
from scipy import stats

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

direc = '/Users/jonty/mydata/robin/revised/'

filelist = np.asarray(os.listdir(direc+'../json_files/'))
filelist = filelist[np.where(filelist != '.DS_Store')]

print(filelist)

#Pawellek 2014 sample
pawellek = ascii.read(direc+'pawellek_2014.txt',guess=False,delimiter=',')

paw_tgt   = pawellek['Name'].data

paw_ls    = np.asarray(pawellek['lstar'].data)
paw_rad   = np.asarray(pawellek['rcold'].data)
paw_sblow = np.asarray(pawellek['sblow'].data)
paw_smin  = np.asarray(pawellek['smin'].data)
paw_usmin = np.asarray(pawellek['usmin'].data)
paw_q     = np.asarray(pawellek['q'].data)
paw_uq    = np.asarray(pawellek['uq'].data)

paw_sratio = paw_smin/paw_sblow

def cornerplot(sampler):
        flat_samples = sampler
        
        if ndim == 4:
            fig = corner.corner(flat_samples, labels=["mdust", "q", "smin", "epsilon"], quantiles=[0.16, 0.5, 0.84],show_titles=True)
        if ndim == 3:
            fig = corner.corner(flat_samples, labels=["mdust", "smin", "epsilon"], quantiles=[0.16, 0.5, 0.84],show_titles=True)

        #plt.suptitle('{:} Corner Plot'.format(target))
        #plt.savefig(direc+'{:}_cornerplot'.format(target))
        fig.show()

targets = []

#fitted model parameters
q     = []
q_ep  = []
q_em  = []
md    = []
md_ep = []
md_em = []
ad    = []
ad_ep = []
ad_em = []

#stellar parameters
dstar = []
lstar = []
mstar = []

#disc temperatures
twarm = []
tcold = []

composition = 'temp2'

nfail = 0
khot = 0
for f in filelist:
    target = f.split('.')[0].strip()
        
    try:
        print('Processing target ',target)

        sdb = open(direc+"../json_files/{:}.json".format(target))
        sdbjson = json.load(sdb)
        
        with open(direc+ 'output_'+composition+'/'+target+'_flatchain.json', 'r') as payload:
            sampler = json.load(payload)
        
        sampler = np.asarray(sampler)
        
        ndim = sampler.shape[1]
        
        if ndim == 4:
            labels=["mdust", "q", "smin", "epsilon"]
            for i in range(0,ndim-1): 
               
                if i == 0:
                    quant = np.quantile(10**sampler[:,i], [0.16, 0.5, 0.84], axis=0)
                    md.append(quant[1])
                    md_ep.append(quant[2] - quant[1])
                    md_em.append(quant[1] - quant[0]) 

                if i == 1:
                    quant = np.quantile(sampler[:,i], [0.16, 0.5, 0.84], axis=0)
                    q.append(quant[1])
                    q_ep.append(quant[2] - quant[1])
                    q_em.append(quant[1] - quant[0])
                   
                if i == 2:
                    quant = np.quantile(10**sampler[:,i], [0.16, 0.5, 0.84], axis=0)
                    ad.append(quant[1])
                    ad_ep.append(quant[2] - quant[1])
                    ad_em.append(quant[1] - quant[0])
            
        if ndim == 3:
            labels=["mdust", "smin", "epsilon"]
            for i in range(0,ndim-1): 
               
                quant = np.quantile(10**sampler[:,i], [0.16, 0.5, 0.84], axis=0)
               
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
        #warm belt
        if len(sdbjson["main_results"]) > 2:
            khot+=1
            t1 = sdbjson["main_results"][1]["Temp"]
            t2 = sdbjson["main_results"][2]["Temp"]
        
            #print(t1,t2)
            #Remove inner warm component from disc - replace with proper filters!
            if t1 < t2:
                ind_warm = 2
                ind_cold = 1
            else:
                ind_warm = 1
                ind_cold = 2
            tcold.append(sdbjson["main_results"][ind_cold]["Temp"])
            twarm.append(sdbjson["main_results"][ind_warm]["Temp"])
            print("Hot component system : ", target, khot)
        else:
            tcold.append(sdbjson["main_results"][1]["Temp"])
            twarm.append(-99)
        
        targets.append(target)
        
    
    except:
        print('Target ',target,' sampler file not found.')
        nfail += 1
        
targets = np.asarray(targets)
dstar = np.asarray(dstar)
lstar = np.asarray(lstar)
mstar = np.asarray(mstar)
twarm = np.asarray(twarm)
tcold = np.asarray(tcold)
q     = abs(np.asarray(q))
q_ep  = np.asarray(q_ep)
q_em  = np.asarray(q_em)
md    = np.asarray(md)
md_ep = np.asarray(md_ep)
md_em = np.asarray(md_em)
ad    = np.asarray(ad)
ad_ep = np.asarray(ad_ep)
ad_em = np.asarray(ad_em)

#Rd vs q
disc_extents = ascii.read(direc+'../targets/'+'resolved_discs_data_table_published_REASONS.csv')

discnam = disc_extents['Target'].data
radii   = disc_extents['R'].data
radii_ep = disc_extents['R_1sigup'].data
radii_em = disc_extents['W_1sigdwn'].data
wides   = disc_extents['W'].data
wides_ep = disc_extents['W_1sigup'].data
wides_em = disc_extents['W_1sigdwn'].data
origins = disc_extents['Origin'].data
pa = disc_extents['pa'].data
inc = disc_extents['inc'].data


disc_lstar = []
disc_name = []
disc_rad = []
disc_rad_ep = []
disc_rad_em = []
disc_wid = []
disc_wid_ep = []
disc_wid_em = []
disc_ori  = []
disc_pa   = []
disc_inc  = []
for i in range(len(discnam)):
    discname= discnam[i]
    if discname in targets:
        index = np.where(discname == targets)[0]
        if discname not in disc_name : 
            disc_name.append(discname)
            disc_lstar.append(lstar[index][0])
            disc_rad.append(radii[i])
            disc_rad_ep.append(radii_ep[i])
            disc_rad_em.append(radii_em[i])
            disc_wid.append(wides[i])
            disc_wid_ep.append(wides_ep[i])
            disc_wid_em.append(wides_em[i])
            disc_ori.append(origins[i])
            disc_pa.append(pa[i])
            disc_inc.append(inc[i])

disc_lstar = np.asarray(disc_lstar)
disc_name = np.asarray(disc_name)
disc_rad = np.asarray(disc_rad)
disc_rad_ep = np.asarray(disc_rad_ep)
disc_rad_em = np.asarray(disc_rad_em)
disc_wid = np.asarray(disc_wid)
disc_wid_ep = np.asarray(disc_wid_ep)
disc_wid_em = np.asarray(disc_wid_em)
disc_ori  = np.asarray(disc_ori)
disc_pa   = np.asarray(disc_pa)
disc_inc  = np.asarray(disc_inc)

ls_sort = np.argsort(disc_lstar)

ls_sort_discnam = disc_name[ls_sort]
ls_sort_radii = disc_rad[ls_sort]
ls_sort_radii_ep = disc_rad_ep[ls_sort]
ls_sort_radii_em = disc_rad_em[ls_sort]
ls_sort_wides = disc_wid[ls_sort]
ls_sort_wides_ep = disc_wid_ep[ls_sort]
ls_sort_wides_em = disc_wid_em[ls_sort]
ls_sort_origins = disc_ori[ls_sort]
ls_sort_pa = disc_pa[ls_sort]
ls_sort_inc = disc_inc[ls_sort]


#lstar vs mdisc
qbig = 2.8
qmed = 3.7
smm  = 3e-3
s1km = 1e3
smax = 2e5
mk   = md * ((4-q)/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)
mk_ep = (md+md_ep) * ((4-(q-q_ep))/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig) - mk
mk_em = mk - (md-md_em) * ((4-(q+q_em))/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)

#3-parameter fits vs. Lstar
rho = 3.3 #gcm^-3
ablow = (0.574/0.5)*(1./rho)*mstar*lstar

#Make Latex Table
ls_sort = np.argsort(lstar)

ls_sort_targets = targets[ls_sort]
ls_sort_lstar = lstar[ls_sort]
ls_sort_mstar = mstar[ls_sort]
ls_sort_dstar = dstar[ls_sort]
ls_sort_q     = q[ls_sort]
ls_sort_q_ep  = q_ep[ls_sort]
ls_sort_q_em  = q_em[ls_sort]
ls_sort_smin    = ad[ls_sort]
ls_sort_smin_ep = ad_ep[ls_sort]
ls_sort_smin_em = ad_em[ls_sort]
ls_sort_md    = md[ls_sort]
ls_sort_md_ep = md_ep[ls_sort]
ls_sort_md_em = md_em[ls_sort]
ls_sort_mk    = mk[ls_sort]
ls_sort_mk_ep = mk_ep[ls_sort]
ls_sort_mk_em = mk_em[ls_sort]
ls_sort_ad    = ad[ls_sort]
ls_sort_ad_ep = ad_ep[ls_sort]
ls_sort_ad_em = ad_em[ls_sort]

ls_sort_sblow = (0.574/0.5)*(1./rho)*ls_sort_mstar*ls_sort_lstar


names = []
lumin_string = []
smass_string = []
radii_string = []
wides_string = []
sblow_string = []
smin_string  = []
q_string     = []
mdust_string = []
mdisc_string = []
f1mm_string  = []
f10mm_string = []

for i in range(len(ls_sort_targets)):
    names.append(ls_sort_targets[i] + ' & ')
    stra = "{0:#.3f}".format(ls_sort_lstar[i])
    lumin_string.append(stra+' & ')
    stra = "{0:#.3f}".format(ls_sort_mstar[i])
    smass_string.append(stra+' & ')
    #radii
    if np.isnan(ls_sort_radii_em[i]):
        stra = "{0:#.0f}".format(int(ls_sort_radii[i]))
        radii_string.append(stra+' &')
    elif np.isfinite(ls_sort_radii_ep[i]) and np.isfinite(ls_sort_radii_em[i]):
        stra = "{0:#.0f}".format(int(ls_sort_radii[i]))
        strb = "{0:#.0f}".format(int(ls_sort_radii_ep[i]))
        strc = "{0:#.0f}".format(int(ls_sort_radii_em[i]))
        radii_string.append('$'+stra+'^{+'+strb+'}_{'+strc+'}$ &')
    else:
        print(ls_sort_targets[i],' mixed radii ',ls_sort_radii[i],ls_sort_radii_ep[i],ls_sort_radii_em[i])
    #widths
    if np.isnan(ls_sort_wides_ep[i]) and np.isnan(ls_sort_wides_em[i]):
        stra = "{0:#.0f}".format(int(ls_sort_wides[i]))
        wides_string.append('$\leq $'+stra+' &')
    elif np.isfinite(ls_sort_wides_ep[i]) and np.isfinite(ls_sort_wides_em[i]):
        stra = "{0:#.0f}".format(ls_sort_wides[i])
        strb = "{0:#.0f}".format(ls_sort_wides_ep[i])
        strc = "{0:#.0f}".format(ls_sort_wides_em[i])
        wides_string.append('$'+stra+'^{+'+strb+'}_{'+strc+'}$ &')
    else:
        print(ls_sort_targets[i],' mixed wides')
    #sblow
    if ls_sort_sblow[i] < 0.01:
        #sblw_fmt = "{0:#.3f}"
        #stra = sblw_fmt.format(ls_sort_sblow[i])
        sblow_string.append('$<$ 0.01 & ')
    elif 0.01 <= ls_sort_sblow[i] <= 0.95:
        sblw_fmt = "{0:#.2f}"
        stra = sblw_fmt.format(ls_sort_sblow[i])
        sblow_string.append(stra+' & ')
    elif 0.95 < ls_sort_sblow[i] < 0.95:
        sblw_fmt = "{0:#.1f}"
        stra = sblw_fmt.format(ls_sort_sblow[i])
        sblow_string.append(stra+' & ')
    elif 0.95 <= ls_sort_sblow[i] < 10.0:
        sblw_fmt = "{0:#.1f}"
        stra = sblw_fmt.format(ls_sort_sblow[i])
        sblow_string.append(stra+' & ')
    elif ls_sort_sblow[i] >= 10.:
        sblw_fmt = "{0:#.0f}"
        stra = sblw_fmt.format(ls_sort_sblow[i])
        sblow_string.append(stra+' & ')
    stra = "{0:#.2f}".format(ls_sort_smin[i])
    strb = "{0:#.2f}".format(ls_sort_smin_ep[i])
    strc = "{0:#.2f}".format(ls_sort_smin_em[i])
    smin_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.1f}".format(ls_sort_q[i])
    strb = "{0:#.1f}".format(ls_sort_q_ep[i])
    strc = "{0:#.1f}".format(ls_sort_q_em[i])
    if ls_sort_q_ep[i] != 0.0 and ls_sort_q_em[i] != 0.0: 
        q_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    else:
        q_string.append('3.5 & ')
    stra = "{0:#.1f}".format(ls_sort_md[i]*1e3)
    strb = "{0:#.1f}".format(ls_sort_md_ep[i]*1e3)
    strc = "{0:#.1f}".format(ls_sort_md_em[i]*1e3)
    mdust_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.1f}".format(ls_sort_mk[i])
    strb = "{0:#.1f}".format(ls_sort_mk_ep[i])
    strc = "{0:#.1f}".format(ls_sort_mk_em[i])
    mdisc_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.3f}".format(0.000)
    strb = "{0:#.3f}".format(1.000)
    strc = "{0:#.3f}".format(1.000)
    f1mm_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.0f}".format(0)
    strb = "{0:#.0f}".format(1)
    strc = "{0:#.0f}".format(1)
    f10mm_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ \\\\ ')

names_string = np.array(names)
lumin_string = np.array(lumin_string)
smass_string = np.array(smass_string)
radii_string = np.array(radii_string)
wides_string = np.array(wides_string)
sblow_string = np.array(sblow_string)
q_string     = np.array(q_string)
mdust_string = np.array(mdust_string)
mdisc_string = np.array(mdisc_string)

predictions = ascii.read(direc + 'extrapolate/emcee_resolved_disc_sed_flux_predictions.dat',delimiter=',')

f1mm_string  = np.array(predictions['FBand6'].data)
f10mm_string = np.array(predictions['FBand1'].data)

ascii.write([names_string,lumin_string,smass_string,radii_string,wides_string,sblow_string,smin_string,q_string,\
             mdust_string,mdisc_string,f1mm_string,f10mm_string],\
             direc+'emcee_resolved_disc_sed_fit_output_latex.dat',\
             names=['Name', 'Lstar', 'Mstar','Radius','Width','sblow','smin','q','Mdust','Mdisk','F1mm','F10mm'],\
             overwrite=True)

ascii.write([ls_sort_targets,ls_sort_lstar,ls_sort_mstar,ls_sort_radii,ls_sort_wides,ls_sort_sblow,ls_sort_smin,ls_sort_smin_ep,ls_sort_smin_em,\
             ls_sort_q,ls_sort_q_ep,ls_sort_q_em,ls_sort_md,ls_sort_md_ep,ls_sort_md_em],\
             direc+'emcee_resolved_disc_tablulated_values_for_modelling.csv',delimiter=',',\
             names=['Name', 'Lstar', 'Mstar','Radius','Width','sblow','smin','smin_ep','smin_em','q','q_ep','q_em','Md','Md_ep','Md_em'],\
             overwrite=True)

ls_sort_f7mm = []
for i in range(len(ls_sort_targets)):
    f10mm_float = float(f10mm_string[i].split('^')[0].strip('$'))
    ls_sort_f7mm.append(f10mm_float)
    if f10mm_float >= 15. :
        print(ls_sort_targets[i],2*ls_sort_radii[i]/ls_sort_dstar[i]),f10mm_float
ls_sort_f7mm = np.asarray(ls_sort_f7mm)

#data table for ALMA proposal
ascii.write([ls_sort_targets,ls_sort_lstar,ls_sort_mstar,ls_sort_dstar,ls_sort_radii,ls_sort_wides,ls_sort_pa,ls_sort_inc,ls_sort_f7mm],\
             direc+'band1_resolved_disc_properties.dat',delimiter=',',\
             names=['Name', 'Lstar', 'Mstar','dstar','Radius','FWHM','pa','inc','f7mm'],\
             overwrite=True)


#Gas rich discs sample
gas_tgt = ['HD9672','HD21997','HD32297','HD39060','HD95086','HD109573','HD110058','HD121191','HD121617','HD131835','HD138813']

gas = []
for gt in gas_tgt:
    gas.append(np.where(targets == gt)[0][0])
print(gas)

gas_ls    = lstar[gas]
gas_sblow = ablow[gas]
gas_smin  = ad[gas]
gas_usmin = 0.5 * (ad_ep[gas] + ad_em[gas])
gas_q     = q[gas]
gas_uq    = 0.5 * (q_ep[gas] + q_em[gas])

realq = np.where(q_ep != 0.0)
fixq = np.where(q_ep == 0.0)

#a_min vs a_blow
fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
lstar_clip = np.clip(np.log10(lstar),-2.0,2.0)
im = ax.scatter(ablow,ad,marker='o',c=lstar_clip,cmap='coolwarm_r',vmin=-2,vmax=2)
#ax.errorbar(ablow[fixq],ad[fixq],xerr=None,yerr=[ad_em[fixq],ad_ep[fixq]],linestyle='',marker='o',c=lstar_clip,cmap='coolwarm_r')
ax.plot([0.01,100.0],[0.01,100.0],linestyle='--',marker='',color='k')
#gas properties
#ax.scatter(gas_sblow,gas_smin,marker='s',c=gas_ls,cmap='coolwarm_r',vmin=-2,vmax=2)
#hatched region denotes realistic blowout sizes
ax.add_patch(Polygon([[0.01, 0.01], [0.37, 0.01], [0.37, 100.0], [0.01, 100.]], closed=True,
                      fill=False, hatch='/',color='grey'))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.01,100.0])
ax.set_ylim([0.01,100.0])
ax.set_xlabel(r'Blowout grain size ($\mu$m)')
ax.set_ylabel(r'Minimum grain size ($\mu$m)')
cbar = plt.colorbar(mappable=im,cmap='coolwarm', label=r'$L_{\star}$ ($L_{\odot}$)',orientation='vertical')
levels = np.linspace(-2, 2, 5)
cbar.set_ticks(levels,labels=[r'10$^{-2}$',r'10$^{-1}$',r'1',r'10$^{1}$',r'10$^{2}$'])
plt.draw()
plt.show()
fig.savefig(direc+'ablow_vs_amin_3parameter_'+composition+'.pdf', dpi=200)
plt.close()

#Rd vs q
disc_extents = ascii.read(direc+'../targets/'+'resolved_discs_data_table.csv')

discnam = disc_extents['Target'].data
radii   = disc_extents['R'].data
wides   = disc_extents['W'].data

#compare cold radii vs resolved extents
rbb_cold = (278.0/tcold)**2.0 * lstar**0.5
rac_cold = []
for i in range(len(targets)):
    index = np.where(targets[i] == discnam)
    #print(targets[i], discnam[index][0])
    rac_cold.append(radii[index][0])

rac_cold = np.asarray(rac_cold)
radii_ratio_cold = rac_cold/rbb_cold

rbb_warm = (278.0/twarm)**2.0 * lstar**(-0.5)
nwarm = np.where(twarm == -99)
rbb_warm[nwarm] = 0.

rac_warm = radii_ratio_cold * rbb_warm

the_warm = rac_warm / dstar


fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(lstar,rac_cold/rbb_cold,xerr=None,yerr=None,linestyle='',marker='o',mec='black',mfc='white')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.01,100.0])
ax.set_ylim([0.1,100.0])
ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
ax.set_ylabel(r'$R_{disc}/R_{bb}$')
plt.show()
fig.savefig(direc+'lstar_vs_radii_ratio_cold_3parameter_'+composition+'.pdf', dpi=200)
plt.close()

mittal = ascii.read(direc+'mittal2015_lstar_rwarm.txt')
mittal_lstar = mittal['Lstar'].data
mittal_rwarm = mittal['Rwarm'].data

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(lstar,rac_warm,xerr=None,yerr=None,linestyle='',marker='o',mec='black',mfc='white')
ax.errorbar(mittal_lstar,mittal_rwarm,xerr=None,yerr=None,linestyle='',marker='o',mec='white',mfc='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.1,100.0])
ax.set_ylim([0.1,30.0])
ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
ax.set_ylabel(r'Predicted radius of warm SED component (au)')
plt.show()
fig.savefig(direc+'lstar_vs_radii_warm_pred_3parameter_'+composition+'.pdf', dpi=200)
plt.close()

#radii for discs with real q values
realq = np.where(ls_sort_q != 3.5)
#alma = np.where((ls_sort_origins != 'ALMA/SMA') & (ls_sort_q != 3.5))
#alma = np.where((ls_sort_origins != 'HERSCHEL') & (ls_sort_q != 3.5))
alma = np.where(ls_sort_q != 3.5)

from lmfit import Model, Parameters
x = np.log10(ls_sort_radii[alma])
y = ls_sort_q[alma]
dy = 0.5*(ls_sort_q_em[alma] + ls_sort_q_ep[alma])/ls_sort_q[alma]

from scipy.stats import rankdata
from scipy.stats import spearmanr,permutation_test
rx = rankdata(x)
ry = rankdata(y)

sigrx = 0.5 * (abs(ls_sort_radii_em[alma]) + abs(ls_sort_radii_ep[alma]))
sigry = 0.5 * (abs(ls_sort_q_em[alma]) + abs(ls_sort_q_ep[alma]))

sigrreal = np.isfinite(sigrx)
#rhospear = np.cov(rx[sigrreal],ry[sigrreal])[0,0] / np.sum((sigrx[sigrreal] * sigry[sigrreal]))

rs = 1. - (6. *  np.sum((rx - ry))**2) / (len(rx)*(len(rx)**2 - 1.))

rhos = spearmanr(rx,ry)

def statistic(x,y): # permute only `x`
    return spearmanr(x,y).statistic
res_exact = permutation_test((x,y), statistic,permutation_type='independent')
res_asymptotic = spearmanr(x[sigrreal], y[sigrreal])
print(res_exact.pvalue, res_asymptotic.pvalue)

def straight_line(t, exp, const):
    #return c* t**m
    return exp*t + const

qs_ep = ls_sort_q_ep[alma]
qs_em = ls_sort_q_em[alma]

tars = ls_sort_targets[alma]

params = Parameters()
params.add('exp', value=0.0)
params.add('const', value=3.5)

dmodel = Model(straight_line)
result = dmodel.fit(y, params, t=x,weights=1./dy)
print(result.fit_report())

mbf  = result.best_values['exp']
msig = result.uvars['exp'].s
cbf  = result.best_values['const']
csig = result.uvars['const'].s

lstar_clip = np.clip(np.log10(lstar[realq]),-2.0,2.0)

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(10**x,y,xerr=None,yerr=[qs_em,qs_ep],linestyle='',marker='o',ms=8,color='black',mec='white',mfc='black',label='other discs')
# ax.errorbar(rq[realqqa],abs(q[realqqa]),xerr=None,yerr=[q_em[realqqa],q_ep[realqqa]],linestyle='',marker='o',color='green',mec='white',mfc='green')
# ax.errorbar(rq[realqqb],abs(q[realqqb]),xerr=None,yerr=[q_em[realqqb],q_ep[realqqb]],linestyle='',marker='o',color='blue',mec='white',mfc='blue')
xmod = np.arange(1,500)
ax.plot(xmod,abs(mbf*np.log10(xmod) + cbf),linestyle='-',color='black')

#uncertainty
mod = cbf + mbf*np.log10(xmod) #cbf*xmod**mbf #
sigmod = abs(mod) * np.sqrt(((mbf/cbf)*csig)**2 + (np.log10(cbf)*msig)**2 )

#1-sigma
#ax.plot(xmod,mod + sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - sigmod,linestyle='-',color='darkgrey')
#2-sigma
#ax.plot(xmod,mod + 2*sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - 2*sigmod,linestyle='-',color='darkgrey')
#3-sigma
#ax.plot(xmod,mod + 3*sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - 3*sigmod,linestyle='-',color='darkgrey')
ax.set_ylim(5.,2.)
#ax.errorbar(paw_rad,paw_q,yerr=paw_uq,linestyle='',marker='s',color='black')

#Overplot random samples from distribution

#theoretical models
ax.add_patch(Polygon([[1.0,3.25], [1.0,3.0], [1000.,3.0], [1000.,3.25]], closed=True,
                      fill=False, hatch='/',color='red'))
ax.add_patch(Polygon([[1.0, 3.38], [1.0, 3.24], [1000., 3.24], [1000., 3.38]], closed=True,
                      fill=False, hatch='\\',color='orange'))
ax.add_patch(Polygon([[1.0, 3.3], [1.0, 3.4], [1000., 3.4], [1000., 3.3]], closed=True,
                      fill=False, hatch='x',color='lightgreen'))
ax.plot([1.0,1.0e3],[3.5,3.5],linestyle='-',color='turquoise')
ax.add_patch(Polygon([[1.0, 4.0], [1.0, 3.65], [1000., 3.65], [1000., 4.0]], closed=True,
                      fill=False, hatch='//',color='blue'))
ax.add_patch(Polygon([[1.0, 3.65], [1.0, 4.32], [1000., 4.32], [1000., 3.65]], closed=True,
                      fill=False, hatch='\\',color='purple'))

#for i in range(0,99):
#    ax.plot(rq[realqq],abs(mrand[i]*np.log10(rq[realqq]) + crand[i]),linestyle='-',color='goldenrod',alpha=0.1)
tgt_chsn = np.asarray(['HD197481','HD22049','HD61005','HD48370','HD107146','HD377','HD104860','HD105','HD10647','HD181327','HD15115','HD109085','HD218396','HD95086','HD32297','HD39060','HD131835','HD131488','HD9672','HD216956','HD109573'])

arg_chsn = []
for i in range(len(tars)):
    if tars[i] in tgt_chsn:
        arg_chsn.append(i)
        

arg_chsn = np.asarray(arg_chsn)

ax.errorbar(10**x[arg_chsn],abs(y[arg_chsn]),xerr=None,yerr=[qs_em[arg_chsn],qs_ep[arg_chsn]],linestyle='',marker='s',ms=8,mec='white',mfc='black',color='black',label='mm-bright discs')

ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim([20,300.0])
ax.set_ylim([5,2])
ax.set_xlabel(r'Disc Radius (au)')
ax.set_ylabel(r'Exponent of particle size distribution (q)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.draw()
plt.show()
fig.savefig(direc+'radius_vs_q_3parameter_w_theory_'+composition+'.pdf', dpi=200)
plt.close()


#Same as above, no theory hatching
fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(10**x,y,xerr=None,yerr=[qs_em,qs_ep],linestyle='',marker='o',ms=8,color='orange',mec='white',mfc='orange',label='other discs')
# ax.errorbar(rq[realqqa],abs(q[realqqa]),xerr=None,yerr=[q_em[realqqa],q_ep[realqqa]],linestyle='',marker='o',color='green',mec='white',mfc='green')
# ax.errorbar(rq[realqqb],abs(q[realqqb]),xerr=None,yerr=[q_em[realqqb],q_ep[realqqb]],linestyle='',marker='o',color='blue',mec='white',mfc='blue')
xmod = np.arange(1,500)
ax.plot(xmod,abs(mbf*np.log10(xmod) + cbf),linestyle='-',color='black')

#uncertainty
mod = abs(mbf*np.log10(xmod) + cbf)
sigmod = abs(mod) * np.sqrt(((mbf/cbf)*csig)**2 + (np.log10(cbf)*msig)**2 )

#1-sigma
#ax.plot(xmod,mod + sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - sigmod,linestyle='-',color='darkgrey')
#2-sigma
#ax.plot(xmod,mod + 2*sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - 2*sigmod,linestyle='-',color='darkgrey')
#3-sigma
#ax.plot(xmod,mod + 3*sigmod,linestyle='-',color='darkgrey')
#ax.plot(xmod,mod - 3*sigmod,linestyle='-',color='darkgrey')
ax.set_ylim(5.,2.)
#ax.errorbar(paw_rad,paw_q,yerr=paw_uq,linestyle='',marker='s',color='black')

#Overplot random samples from distribution

#theoretical models
#ax.add_patch(Polygon([[1.0,3.25], [1.0,3.0], [1000.,3.0], [1000.,3.25]], closed=True,
#                      fill=False, hatch='/',color='red'))
#ax.add_patch(Polygon([[1.0, 3.38], [1.0, 3.24], [1000., 3.24], [1000., 3.38]], closed=True,
#                      fill=False, hatch='\\',color='orange'))
#ax.add_patch(Polygon([[1.0, 3.3], [1.0, 3.4], [1000., 3.4], [1000., 3.3]], closed=True,
#                      fill=False, hatch='x',color='lightgreen'))
#ax.plot([1.0,1.0e3],[3.5,3.5],linestyle='-',color='turquoise')
#ax.add_patch(Polygon([[1.0, 4.0], [1.0, 3.65], [1000., 3.65], [1000., 4.0]], closed=True,
#                      fill=False, hatch='//',color='blue'))
#ax.add_patch(Polygon([[1.0, 3.65], [1.0, 4.32], [1000., 4.32], [1000., 3.65]], closed=True,
#                      fill=False, hatch='\\',color='purple'))

#for i in range(0,99):
#    ax.plot(rq[realqq],abs(mrand[i]*np.log10(rq[realqq]) + crand[i]),linestyle='-',color='goldenrod',alpha=0.1)
tgt_chsn = np.asarray(['HD197481','HD22049','HD61005','HD48370','HD107146','HD377','HD104860','HD105','HD10647','HD181327','HD15115','HD109085','HD218396','HD95086','HD32297','HD39060','HD131835','HD131488','HD9672','HD216956','HD109573'])

arg_chsn = []
for i in range(len(tars)):
    if tars[i] in tgt_chsn:
        arg_chsn.append(i)
        

arg_chsn = np.asarray(arg_chsn)

ax.errorbar(10**x[arg_chsn],abs(y[arg_chsn]),xerr=None,yerr=[qs_em[arg_chsn],qs_ep[arg_chsn]],linestyle='',marker='s',ms=8,mec='white',mfc='dodgerblue',color='dodgerblue',label='mm-bright discs')

ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim([20,300.0])
ax.set_ylim([5,2])
ax.set_xlabel(r'Disc Radius (au)')
ax.set_ylabel(r'Exponent of particle size distribution (q)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.draw()
plt.show()
fig.savefig(direc+'radius_vs_q_3parameter_wout_theory_'+composition+'.pdf', dpi=200)
plt.close()

#Convert Radius to Keplerian velocity
import astropy.constants as const
import astropy.units as units
constG    = const.G
constMsun = const.M_sun
constAU   = const.au

rm = ls_sort_mstar[realq]
qs = ls_sort_q[realq]

vkepler = np.sqrt( (constG * constMsun * ls_sort_mstar[realq])/ (ls_sort_radii[realq]*constAU))
lsort   = ls_sort_lstar[realq]
xsort   = ls_sort_radii[realq]
nsort   = ls_sort_discnam[realq]

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
lstar_clip = np.clip(np.log10(lsort),-2.0,2.0)
ax.errorbar(vkepler/1e3,qs,xerr=None,yerr=[qs_em,qs_ep],linestyle='',marker='.',color='k',zorder=10)
im = ax.scatter(vkepler/1e3,qs,marker='o',c=lstar_clip,cmap='coolwarm_r',vmin=-2,vmax=2,zorder=19)
#ax.errorbar(ablow[fixq],ad[fixq],xerr=None,yerr=[ad_em[fixq],ad_ep[fixq]],linestyle='',marker='o',c=lstar_clip,cmap='coolwarm_r')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlim([1.0,8.0])
ax.set_ylim([2.0,5.0])
ax.set_xlabel(r'Keplerian Velocity ($v_{\rm Kep}$, km/s)')
ax.set_ylabel(r'Size Distribution Exponent ($q$)')
cbar = plt.colorbar(mappable=im,cmap='coolwarm', label=r'$L_{\star}$ ($L_{\odot}$)',orientation='vertical')
levels = np.linspace(-2, 2, 5)
cbar.set_ticks(levels,labels=[r'10$^{-2}$',r'10$^{-1}$',r'1',r'10$^{1}$',r'10$^{2}$'])
plt.draw()
plt.show()
fig.savefig(direc+'q_vs_vkep_3parameter_'+composition+'.pdf', dpi=200)
plt.close()


#lstar vs mdisc
qbig = 2.8
qmed = 3.7
smm  = 3e-3
s1km = 1e3
smax = 2e5
mdisk = ls_sort_md * ((4-ls_sort_q)/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(ls_sort_mstar[realq],mdisk[realq],xerr=None,yerr=None,linestyle='',marker='o',color='black',mec='white',mfc='black')
ax.errorbar(ls_sort_mstar[fixq],mdisk[fixq],xerr=None,yerr=None,linestyle='',marker='o',color='black',mec='white',mfc='darkgrey')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.3,3.0])
ax.set_ylim([1e-2,1e4])
ax.set_xlabel(r'Stellar Mass ($M_{\odot}$)')
ax.set_ylabel(r'Total Disc Mass ($M_{\oplus}$)')
ax.set_xticks([0.3,0.5,1.0,2.0,3.0])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()
plt.show()
fig.savefig(direc+'mstar_vs_mdisc_3parameter_'+composition+'.pdf', dpi=200)
plt.close()

#trend between stellar mass and disk mass? - NO!
# params = Parameters()
# params.add('m', value=0.0)
# params.add('c', value=3e3)

# x = mstar[realq]
# y = mdisk[realq]

# dmodel = Model(straight_line)
# result = dmodel.fit(y, params, t=x)
# print(result.fit_report())
# [[Model]]
#     Model(straight_line)
# [[Fit Statistics]]
#     # fitting method   = leastsq
#     # function evals   = 7
#     # data points      = 80
#     # variables        = 2
#     chi-square         = 3.4344e+10
#     reduced chi-square = 4.4031e+08
#     Akaike info crit   = 1594.21438
#     Bayesian info crit = 1598.97843
#     R-squared          = 0.00189266
# [[Variables]]
#     m:  4526.59832 +/- 11770.0289 (260.02%) (init = 0)
#     c:  4290.67515 +/- 3065.20005 (71.44%) (init = 3000)
# [[Correlations]] (unreported correlations are < 0.100)
#     C(m, c) = -0.6436

#lstar vs q

#create histogram from posteriors
nhistbin = 125
hist_x   = 1.95+((0.05*np.arange(0,nhistbin)) + 0.025)
hist_list = np.asarray([])

from scipy.interpolate import interp1d

for f in filelist:
    target = f.split('.')[0].strip()
    
    #print('Processing target ',target)
    
    sdb = open(direc+"../json_files/{:}.json".format(target))
    sdbjson = json.load(sdb)
    
    swave = sdbjson['star_spec']['wavelength']
    sphot = np.asarray(sdbjson['star_spec']['fnujy'])*1e6
    
    photosphere = interp1d(swave,sphot)
    fstar_7mm = photosphere(7e3)
    #print(target,fstar_7mm)
    
    with open(direc+ 'output_'+composition+'/'+target+'_flatchain.json', 'r') as payload:
        sampler = json.load(payload)
    
    sampler = np.asarray(sampler)
    
    ndim = sampler.shape[1]

    if ndim == 4:
        labels=["mdust", "q", "smin", "epsilon"]
        
        hist = np.histogram(abs(sampler[7999::,1]),bins=nhistbin,range=(1.95,5.05))
        
        hist_list = np.append(hist_list, abs(sampler[7999::,1]))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
#ax.set_ylabel(r'Stellar Luminosity ($L_{\odot}$)')
ax.set_ylabel(r'Normalized Probability',fontsize=24)
ax.set_xlabel(r'Exponent of particle size distribution ($q$)',fontsize=24)
ax.set_xlim(1.95,5.05)
#ax.set_ylim(0.01,100.0)
ax.set_ylim(0.0,5e5)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.invert_xaxis()

ax.add_patch(Polygon([[3.25, 0.0], [3.0, 0.0], [3.0, 1.0e6], [3.25, 1.0e6]], closed=True,
                      fill=False, hatch='/',color='red',label=r'Pan & Sari 2005'))
ax.add_patch(Polygon([[3.38, 0.0], [3.24, 0.0], [3.24, 1.0e6], [3.38, 1.0e6]], closed=True,
                      fill=False, hatch='\\',color='orange',label=r'Norfolk et al. 2021'))
ax.add_patch(Polygon([[3.3, 0.0], [3.4, 0.0], [3.4, 1.0e6], [3.3, 1.0e6]], closed=True,
                      fill=False, hatch='x',color='lightgreen',label=r'Schuppler et al. 2015'))
ax.plot([3.5,3.5],[0.0,1.0e6],linestyle='-',color='turquoise',label=r'Dohnanyi 1969')
ax.add_patch(Polygon([[4.0, 0.0], [3.65, 0.0], [3.65, 1.0e6], [4.0, 1.0e6]], closed=True,
                      fill=False, hatch='//',color='blue',label=r'Pan & Schlichting 2012'))
ax.add_patch(Polygon([[3.60, 0.0], [3.70, 0.0], [3.70, 1.0e6], [3.60, 1.0e6]], closed=True,
                      fill=False, hatch='\\',color='purple',label=r'Gaspar et al. 2012'))
#ax.errorbar(ls_sort_q[realq],ls_sort_lstar[realq],xerr=[ls_sort_q_em[realq],ls_sort_q_ep[realq]],marker='o',linestyle='',color='black',mec='white',mfc='black')
ax.hist(hist_list,bins=nhistbin,histtype='bar',color='k',alpha=0.2)
ax.legend(loc='upper left',fontsize=12)

ax.set_yticks([0.0,250000,500000])
ax.set_yticklabels(['0.0','0.01','0.02'],fontsize=18)
ax.set_xticks([5.0,4.5,4.0,3.5,3.0,2.5,2.0])
ax.set_xticklabels(['5.0','4.5','4.0','3.5','3.0','2.5','2.0'],fontsize=18)

plt.tight_layout()
plt.show()
fig.savefig(direc+'q_histogram_'+composition+'.pdf', dpi=200)
plt.close()

#a_min
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_lstar[realq],ls_sort_ad[realq],xerr=None,yerr=[ls_sort_ad_em[realq],ls_sort_ad_ep[realq]],linestyle='',marker='o',color='firebrick',mec='white',mfc='firebrick')
plt.errorbar(ls_sort_lstar[fixq],ls_sort_ad[fixq],xerr=None,yerr=[ls_sort_ad_em[fixq],ls_sort_ad_ep[fixq]],linestyle='',marker='o',color='firebrick',mec='firebrick',mfc='white')
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.001,100.0))
plt.ylim((0.01,100.0))
plt.xlabel(r'Stellar luminosity ($L_{\odot}$)')
plt.ylabel(r'Minimum grain size $s_{\rm min}$ ($\mu$m)')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_amin_3parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(ls_sort_lstar[realq]),np.log10(ls_sort_ad[realq]))

#a_min/a_blow
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)

herc = np.where((ls_sort_origins == 'HERSCHEL') & (ls_sort_q != 3.5))
alma = np.where((ls_sort_origins != 'HERSCHEL') & (ls_sort_q != 3.5))

ax.errorbar(ls_sort_lstar[herc],ls_sort_ad[herc]/ls_sort_sblow[herc],xerr=None,yerr=[ls_sort_ad_em[herc]/ls_sort_sblow[herc],ls_sort_ad_ep[herc]/ls_sort_sblow[herc]],linestyle='',marker='o',color='dodgerblue',mec='white',mfc='dodgerblue',label='Herschel')
ax.errorbar(ls_sort_lstar[alma],ls_sort_ad[alma]/ls_sort_sblow[alma],xerr=None,yerr=[ls_sort_ad_em[alma]/ls_sort_sblow[alma],ls_sort_ad_ep[alma]/ls_sort_sblow[alma]],linestyle='',marker='o',color='orange',mec='white',mfc='orange',label='ALMA/SMA')
#ax.errorbar(ls_sort_lstar[fixq],ls_sort_ad[fixq]/ls_sort_sblow[fixq],xerr=None,yerr=[ls_sort_ad_em[fixq]/ls_sort_sblow[fixq],ls_sort_ad_ep[fixq]/ls_sort_sblow[fixq]],linestyle='',marker='o',color='black',mec='black',mfc='white',label='Fixed')
#Add shaded regions in here below 1.05 Lo and between 1.05 and 6.15 Lo for the dust grain blowout limits
ax.set_xscale('log')
ax.set_yscale('log')

#Pawellek 2014 relation
paw_rel_lstar = np.logspace(-3,3,num=100,endpoint=True,base=10.0)
paw_rel_ratio = 8.6 * paw_rel_lstar**(-0.47)
prlo = np.where(paw_rel_lstar < 1.06)
prhi = np.where(paw_rel_lstar >= 1.06)
ax.plot(paw_rel_lstar[prhi],paw_rel_ratio[prhi],linestyle='-',color='darkgrey',label='Pawellek+ (2014)')
ax.plot(paw_rel_lstar[prlo],paw_rel_ratio[prlo],linestyle='--',color='darkgrey')

#Fit to L > 1.06 Lsun here
from scipy import stats
lshi = np.where(ls_sort_lstar >= 1.06)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(ls_sort_lstar[lshi]),np.log10(ls_sort_ad[lshi]/ls_sort_sblow[lshi]))

our_rel_ratio = 10**0.86 * paw_rel_lstar**(-1.065)

ax.plot(paw_rel_lstar[prhi],our_rel_ratio[prhi],linestyle='-',color='black',label='This work')
ax.plot(paw_rel_lstar[prlo],our_rel_ratio[prlo],linestyle='--',color='black')

ax.add_patch(Polygon([[0.01, 0.01], [0.01, 100.], [1.06,100.], [1.06,0.01]], closed=True,
                      fill=True,color='lightgrey',alpha=0.2))
ax.add_patch(Polygon([[1.06, 0.01], [1.06, 100.], [6.14, 100.], [6.14, 0.01]], closed=True,
                      fill=False, hatch='\\',color='darkgrey',alpha=0.4))
ax.set_xlim((0.01,100.0))
ax.set_ylim((0.01,100.0))
ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
ax.set_ylabel(r'$s_{\rm min}/s_{\rm blow}$')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_sratio_free_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)

fixq = np.where(ls_sort_q == 3.5)

#ax.errorbar(ls_sort_lstar_realq[herc],ls_sort_sratio_realq[herc],xerr=None,yerr=[ls_sort_sratio_em_realq[herc],ls_sort_sratio_ep_realq[herc]],linestyle='',marker='o',color='dodgerblue',mec='white',mfc='dodgerblue',label='Herschel')
#ax.errorbar(ls_sort_lstar_realq[alma],ls_sort_sratio_realq[alma],xerr=None,yerr=[ls_sort_sratio_em_realq[alma],ls_sort_sratio_ep_realq[alma]],linestyle='',marker='o',color='orange',mec='white',mfc='orange',label='ALMA/SMA')
ax.errorbar(ls_sort_lstar[fixq],ls_sort_ad[fixq]/ls_sort_sblow[fixq],xerr=None,yerr=[ls_sort_ad_em[fixq]/ls_sort_sblow[fixq],ls_sort_ad_ep[fixq]/ls_sort_sblow[fixq]],linestyle='',marker='o',color='black',mec='black',mfc='white',label='Fixed')
#Add shaded regions in here below 1.05 Lo and between 1.05 and 6.15 Lo for the dust grain blowout limits
ax.set_xscale('log')
ax.set_yscale('log')
ax.add_patch(Polygon([[0.01, 0.01], [0.01, 100.], [1.06,100.], [1.06,0.01]], closed=True,
                      fill=True,color='lightgrey',alpha=0.2))
ax.add_patch(Polygon([[1.06, 0.01], [1.06, 100.], [6.14, 100.], [6.14, 0.01]], closed=True,
                      fill=False, hatch='\\',color='darkgrey',alpha=0.4))

ax.plot(paw_rel_lstar[prhi],paw_rel_ratio[prhi],linestyle='-',color='darkgrey',label='Pawellek+ (2014)')
ax.plot(paw_rel_lstar[prlo],paw_rel_ratio[prlo],linestyle='--',color='darkgrey')

ax.plot(paw_rel_lstar[prhi],our_rel_ratio[prhi],linestyle='-',color='black',label='This work')
ax.plot(paw_rel_lstar[prlo],our_rel_ratio[prlo],linestyle='--',color='black')

ax.set_xlim((0.01,100.0))
ax.set_ylim((0.01,100.0))
ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
ax.set_ylabel(r'$s_{\rm min}/s_{\rm blow}$')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_sratio_fixed_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

#a_min/a_blow vs q
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_q[realq],ls_sort_ad[realq]/ls_sort_sblow[realq],xerr=None,yerr=[ls_sort_ad_em[realq]/ls_sort_sblow[realq],ls_sort_ad_ep[realq]/ls_sort_sblow[realq]],linestyle='',marker='o',color='black',mec='white',mfc='black')
#plt.errorbar(q[fixq],ad[fixq]/ablow[fixq],xerr=None,yerr=[ad_em[fixq]/ablow[fixq],ad_ep[fixq]/ablow[fixq]],linestyle='',marker='o',color='firebrick',mec='firebrick',mfc='white')
#Add shaded regions in here below 1.05 Lo and between 1.05 and 6.15 Lo for the dust grain blowout limits
plt.xscale('linear')
plt.yscale('log')
plt.xlim((5.0,2.0))
plt.ylim((0.01,100.0))
plt.xlabel(r'q (-)')
plt.ylabel(r'Ratio of $s_{\rm min}/s_{\rm blow}$')
plt.tight_layout()
plt.savefig(direc+'q_vs_aratio_3parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

#mdust
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_lstar[realq],1e3*ls_sort_md[realq],xerr=None,yerr=[1e3*ls_sort_md_em[realq],1e3*ls_sort_md_ep[realq]],linestyle='',marker='o',color='darkorange',mec='white',mfc='darkorange')
plt.errorbar(ls_sort_lstar[fixq],1e3*ls_sort_md[fixq],xerr=None,yerr=[1e3*ls_sort_md_em[fixq],1e3*ls_sort_md_ep[fixq]],linestyle='',marker='o',color='darkorange',mec='darkorange',mfc='white')
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.001,100.0))
plt.ylim((0.1,100.0))
plt.xlabel(r'Stellar luminosity ($L_{\odot}$)')
plt.ylabel(r'Dust mass ($10^{-3} M_{\oplus}$)')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_mdust_3parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()


#q
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_lstar[realq],ls_sort_q[realq],xerr=None,yerr=[ls_sort_q_em[realq],ls_sort_q_ep[realq]],linestyle='',marker='o',color='dodgerblue',mec='white',mfc='dodgerblue')
#plt.errorbar(paw_ls,paw_q,yerr=paw_uq,linestyle='',marker='s',color='black')
#plt.errorbar(gas_ls,gas_q,yerr=gas_uq,linestyle='',marker='s',color='black')
plt.xscale('log')
plt.yscale('linear')
plt.xlim((0.001,100.0))
plt.ylim((5.0,2.0))
plt.xlabel(r'Stellar luminosity ($L_{\odot}$)')
plt.ylabel(r'$q$ (-)')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_q_3parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()


#q vs a
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_ad[realq],ls_sort_q[realq],xerr=[ls_sort_ad_em[realq],ls_sort_ad_ep[realq]],yerr=[ls_sort_q_em[realq],ls_sort_q_ep[realq]],linestyle='',marker='o',color='green',mec='white',mfc='green')
#plt.errorbar(paw_smin,paw_q,yerr=paw_uq,linestyle='',marker='s',color='black')
#plt.errorbar(gas_smin,gas_q,yerr=gas_uq,linestyle='',marker='s',color='black')
plt.xscale('log')
plt.yscale('linear')
plt.xlim((0.01,100.0))
plt.ylim((5.0,2.0))
plt.xlabel(r'Minimum grain size $s_{\rm min}$ ($\mu$m)')
plt.ylabel(r'$q$ (-)')
plt.tight_layout()
plt.savefig(direc+'amin_vs_q_3parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

#2-parameter fits

#a_min
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_lstar[fixq],ls_sort_ad[fixq],xerr=None,yerr=[ls_sort_ad_em[fixq],ls_sort_ad_ep[fixq]],linestyle='',marker='o',color='blue',mec='white',mfc='blue')
#plt.errorbar(paw_ls,paw_smin,yerr=paw_usmin,linestyle='',marker='s',color='black')
plt.errorbar(gas_ls,gas_smin,yerr=gas_usmin,linestyle='',marker='s',color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.001,100.0))
plt.ylim((0.01,100.0))
plt.xlabel(r'Stellar luminosity ($L_{\odot}$)')
plt.ylabel(r'Minimum grain size $s_{\rm min}$ ($\mu$m)')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_amin_2parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()

#mdust
plt.subplots(figsize=(8,6))
plt.errorbar(ls_sort_lstar[fixq],1e3*ls_sort_md[fixq],xerr=None,yerr=[1e3*ls_sort_md_em[fixq],1e3*ls_sort_md_ep[fixq]],linestyle='',marker='o',color='purple',mec='white',mfc='purple')
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.001,100.0))
plt.ylim((0.1,100.0))
plt.xlabel(r'Stellar luminosity ($L_{\odot}$)')
plt.ylabel(r'Dust mass ($10^{-3} M_{\oplus}$)')
plt.tight_layout()
plt.savefig(direc+'lstar_vs_mdust_2parameter_'+composition+'.pdf', dpi=200)
plt.show()
plt.close()


#Beta vs a_min for astronomical silicate
#Lstar for no blowout = 0.06 LSun
#Lstar for range of blowout = 0.06 - 8 LSun

# import miepython as mpy
# from scipy.interpolate import interp1d

# lmod = np.logspace(-2, 2,num=100,endpoint=True)
# amod = np.logspace(-2, 2,num=100,endpoint=True)

# mmod = lmod**(1./3.5)
# rmod = lmod**(1./3.5)
# tmod = 5784. * (lmod/rmod**2)**(1/4.)

# optconsts = ascii.read(direc+'../emcee/astrosil.lnk')

# as_l = np.asarray(optconsts['col1'].data)
# as_n = np.asarray(optconsts['col2'].data)
# as_k = np.asarray(optconsts['col3'].data)

# fn = interp1d(as_l,as_n)
# fk = interp1d(as_l,as_k)


# plt.subplots(figsize=(8,6))

# lam_wien = 2898. / tmod

# x = 2.*np.pi*amod/lam_wien
# as_nk = np.ones(len(lmod)) * (fn(lam_wien) - 1j*fk(lam_wien))

# qext, qsca, qback, g = mpy.ez_mie(as_nk,x,np.pi/2.)

# beta = 0.574 * (lmod/mmod) * (qext - qsca*np.cos(np.pi*15./180.)) * (1./(3.3*amod))

# colors = ['r']*len(lmod)

# colors = np.asarray(colors)

# lgreen = np.where((beta < 0.5)&(np.max(beta) < 0.5))
# lblue  = np.where((beta < 0.5)&(np.max(beta) > 0.5))

# colors[lgreen] = 'g'
# colors[lblue]  = 'b'

# for i in range(len(beta)):
#     plt.plot(amod[i],beta[i],linestyle='',color=colors[i])

# plt.xscale('log')
# plt.yscale('log')
# plt.xlim((0.01,100.0))
# plt.ylim((0.001,100.0))
# plt.plot([0.01,100.0],[0.5,0.5],linestyle='--',color='black')
# plt.xlabel(r'Grain size ($\mu$m)')
# plt.ylabel(r'Beta')
# plt.tight_layout()
# plt.show()
# # plt.savefig(direc+'lstar_vs_beta_model_'+composition+'.pdf', dpi=200)
# # plt.close()

# print(np.max(lgreen))
# print(np.max(lblue))

#q vs v_rel

ls_sort_period = 2.*np.pi*np.sqrt((ls_sort_radii*const.au)**3 / (const.G * ls_sort_mstar * const.M_sun))

tvals = np.asarray(["HD197481","GJ14","HD9672","HD10647","HD14055","HD15115","HD32297","HD35841","HD50571","HD61005","HD92945","HD110058","HD109573","HD158352","HD161868","HD191089","HD16743"])
hvals = np.asarray([0.02,0.05,0.05,0.04,0.10,0.05,0.08,0.148,0.11,0.04,0.04,0.21,0.05,0.18,0.152,0.19,0.13])
ehvals = np.asarray([0.002,0.03,0.007,0.007,0.0,0.007,0.01,0.05,0.02,0.03,0.01,0.03,0.003,0.02,0.03,0.0,0.015])

#find q vals for the vertically resolved discs

#qvals = np.asarray([3.1,3.4,3.6,3.6,3.4,3.4,3.3,3.1,3.6,3.5,3.8,3.4,3.5,3.3,3.6,3.2,3.7])
#eqvals = np.asarray([0.1,0.2,0.15,0.1,0.1,0.25,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.2,0.1,0.03])

qvals = []
qvals_ep = []
qvals_em = []

vrels = []
vrels_e = []
for i in range(0,len(tvals)):
    for j in range(0,len(ls_sort_targets)):
        if tvals[i] == targets[j]:
            qvals.append(q[j])
            qvals_ep.append(q_ep[j])
            qvals_em.append(q_em[j])
            
            vrelative = (4.*hvals[i]*ls_sort_radii[j]*const.au)/ls_sort_period[j]
            vrels.append(vrelative.value)
            vrels_e.append((ehvals[i]/hvals[i]) * vrelative.value)

qvals = np.asarray(qvals)
eqvals_ep = np.asarray(qvals_ep)
eqvals_em = np.asarray(qvals_em)

vrels = np.asarray(vrels)
vrels_e = np.asarray(vrels_e)

#Edit HD16743's values
qvals[-1] = 3.7
eqvals_em[-1] = 0.03
eqvals_ep[-1] = 0.03


gd = np.where(ehvals != 0.0)
bd = np.where(ehvals == 0.0)

plt.subplots(figsize=(8,6))
plt.xscale('linear')
plt.yscale('linear')
plt.xlim((0.0,500.0))
plt.ylim((4.0,3.0))
plt.errorbar(vrels[gd],qvals[gd],xerr=vrels_e[gd],yerr=[eqvals_em[gd],eqvals_ep[gd]],linestyle='',marker='o',color='black')
plt.errorbar(vrels[bd],qvals[bd],xerr=None,yerr=[eqvals_em[bd],eqvals_ep[bd]],linestyle='',marker='<',mec='black',mfc='white',markersize=8,color='black')
plt.xlabel(r'Relative velocity (m/s)')
plt.ylabel(r'Exponent of particle size distribution ($q$)')
plt.tight_layout()
plt.savefig(direc+'q_vs_vrel_3parameter_'+composition+'.pdf', dpi=200)
plt.show()

#smin vs q
gdq = np.where(ls_sort_q != 3.5)

plt.subplots(figsize=(8,6))
plt.xscale('log')
plt.yscale('linear')
plt.xlim((1e-2,1e2))
plt.ylim((5.0,2.0))
plt.errorbar(ls_sort_ad[gdq],ls_sort_q[gdq],xerr=[ls_sort_ad_em[gdq],ls_sort_ad_ep[gdq]],yerr=[ls_sort_q_em[gdq],ls_sort_q_ep[gdq]],linestyle='',marker='o',color='black')
plt.xlabel(r'Minimum grain size $s_{\rm min}$ ($\mu$m)')
plt.ylabel(r'Exponent of particle size distribution ($q$)')
plt.tight_layout()
plt.savefig(direc+'q_vs_smin_3parameter_'+composition+'.pdf', dpi=200)
plt.show()

params = Parameters()
params.add('exp', value=0.0)
params.add('const', value=3.5)

y = ls_sort_q[gdq]
x = ls_sort_ad[gdq]
dy = 0.5*(ls_sort_q_ep[gdq] + ls_sort_q_em[gdq])

dmodel = Model(straight_line)
result = dmodel.fit(y, params, t=x,weights=1./dy)
print(result.fit_report())

