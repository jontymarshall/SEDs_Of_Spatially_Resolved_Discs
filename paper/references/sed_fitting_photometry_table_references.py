#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:29:55 2025

@author: jonty
"""

import os
import numpy as np
import json
import pandas as pd
from astropy.table import Table
from astropy.io import ascii

direc = '/Users/jonty/mydata/robin/revised/'

#targets
filelist = np.asarray(os.listdir(direc+'../json_files/'))
filelist = filelist[np.where(filelist != '.DS_Store')]

ref_dict = {'BT' : '2000A&A...355L..27H', #Tycho B
            'VT' : '2000A&A...355L..27H', #Tycho V
            'GAIA.G' : '2021A&A...649A...1G', #Gaia G
            '2MJ' : '2006AJ....131.1163S', #2MASS J
            '2MH' : '2006AJ....131.1163S', #2MASS H
            '2MKS' : '2006AJ....131.1163S', #2MASS Ks
            '2MR1J' : '2006AJ....131.1163S', #2MASS J
            '2MR1H' : '2006AJ....131.1163S', #2MASS H
            '2MR1KS' : '2006AJ....131.1163S', #2MASS Ks
            'WISE3P4' : '2010AJ....140.1868W', #WISE W1
            'WISE4P6' : '2010AJ....140.1868W', #WISE W2
            'WISE12' : '2010AJ....140.1868W', #WISE W3
            'WISE22' : '2010AJ....140.1868W', #WISE W4
            'AKARI9' : '2010A&A...514A...1I', #AKARI IRC9
            'AKARI18' : '2010A&A...514A...1I', #AKARI IRC18
            'MIPS24' : '2014ApJS..211...25C', #Spitzer MIPS24
            'MIPS70' : '2014ApJS..211...25C', #Spitzer MIPS70
            'PACS70' : '2024A%26A...688A.203M', #Herschel PACS70
            'PACS100' : '2024A%26A...688A.203M', #Herschel PACS100
            'PACS160' : '2024A%26A...688A.203M', #Herschel PACS160
            'SPIRE250' : '2024yCat.8112....0H', #Herschel SPIRE250
            'SPIRE350' : '2024yCat.8112....0H', #Herschel SPIRE350
            'SPIRE500' : '2024yCat.8112....0H', #Herschel SPIRE500
            'WAV450' : '2017MNRAS.470.3606H', #JCMT SCUBA2 450
            'WAV850' : '2017MNRAS.470.3606H', #JCMT SCUBA2 850
            # 'WAV870' : '', #ALMA / SMA?
            # 'WAV1300' : '', #ALMA / SMA?
            'WAV1200' : '2025A&A...693A.151M'} #ALMA / SMA?

#Origin papers for almost all sources in the sample

dunes_data    = ascii.read(direc+'../targets/DUNES_Eiroa2013.tbl',comment='#',delimiter='&')
dunes         = []
for i in range(0,len(dunes)):
    
    dunes.append('HD'+str(dunes_data['HD'].data[i]))

dunes = np.asarray(dunes)

debris_a_data = ascii.read(direc+'../targets/DEBRIS_A_Thureau2014.tbl',comment='#',delimiter=',')
debris_a      = debris_a_data['ID'].data

debris_fgk_data = ascii.read(direc+'../targets/DEBRIS_FGK_Sibthorpe2018.tbl',comment='#',delimiter='&',guess=False,fast_reader=False)
debris_fgk    = debris_fgk_data['Name'].data

morales_16_data = ascii.read(direc+'../targets/Morales2016.tbl',comment='#',delimiter=' ',guess=False,fast_reader=False)
morales_16    = morales_16_data['Name'].data

marshall_21_data   = ascii.read(direc+'../targets/Marshall+2021_TableB1_edited_HD_or_GJ_Names.csv',comment='#',delimiter=',')
marshall_21   = marshall_21_data['Target'].data

#Spitzer references
chen_14_data = ascii.read(direc+'../targets/SpitzerIRS_Data_Chen2014.tsv',comment='#',delimiter=';')
chen_14_refs = ascii.read(direc+'../targets/SpitzerIRS_Refs_Chen2014.tsv',comment='#',delimiter=';')

trilling_07_data = ascii.read(direc+'../targets/Trilling2007.tbl',comment='#',delimiter='&',guess=False,fast_reader=False)
trilling_07 = trilling_07_data['Name'].data

# bryden_09_data = ascii.read(direc+'../targets/Bryden2009.tbl',comment='#',delimiter=';')
bryden_09 = ['HD10647','HD52265']#bryden_09_data['Name'].data

morales_09 = ['HD10939','HD30422']

carpenter_09 = ['HD138813']

su_06 = ['HD11413','HD216956','HD172167','HD39060']

beichman_06 = ['HD38858']

kospal_09 = ['GJ581','HD22049','HD20794','HD192263','HD48682','HD17925']

#hillenbrand_08 = ['2008ApJ...677..630H']

trilling_08 = ['HD107146','HD128311']

#JCMT SCUBA2 SONS Legacy Program
sons_data = ascii.read(direc+'../targets/SONS_Holland2017.tbl',comment='#',delimiter='&',guess=False,fast_reader=False)
sons    = sons_data['Name'].data

#Steele+ 2016
steele_16 = ['HD377','HD8907','HD61005','HD104860','HD10746']

#ALMA REASONS
dataframe = direc + '../targets/REASONS_DataFrame_withsdbinfo' 
reasons_data = pd.read_pickle(dataframe)

reasons = reasons_data['Target'].to_numpy()


#Loop over all targets
photometry_references = {}

for f in filelist:
    target = f.split('.')[0].strip()
    
    print('Processing target ',target)
    
    phot_bands = [] 
    phot_refs  = []
    
    sdb = open(direc+"../json_files/{:}.json".format(target))
    sdbjson = json.load(sdb)
    
    snr     = np.asarray(sdbjson['phot_fnujy']) / np.asarray(sdbjson['phot_e_fnujy'])
    ignore  = np.asarray(sdbjson['phot_ignore'])
    upperlim = np.asarray(sdbjson['phot_upperlim'])
    
    include = np.where((ignore == False)&(upperlim == False)&(snr >= 3.))
    
    filters = np.asarray(sdbjson['phot_band'])
    filters = filters[include]
    
    #Optical
    if 'BT' in filters:
        
        phot_bands.append('BT')
        phot_refs.append(ref_dict['BT'])
        
    if 'VT' in filters:
        
        phot_bands.append('VT')
        phot_refs.append(ref_dict['VT'])
    
    if 'GAIA.G' in filters: 
        
        phot_bands.append('GAIA.G')
        phot_refs.append(ref_dict['GAIA.G'])
    
    #Near-infrared (2MASS)
    if '2MJ' or '2MR1J' in filters:
        
        phot_bands.append('2MASSJ')
        
        try:
            
            phot_refs.append(ref_dict['2MJ'])
            
        except:
            
            phot_refs.append(ref_dict['2MR1J'])
    
    if '2MH' or '2MR1H' in filters:
        
        phot_bands.append('2MASSH')
        
        try:
            phot_refs.append(ref_dict['2MH'])
            
        except:
            phot_refs.append(ref_dict['2MR1H'])
    
    if '2MJ' or '2MR1J' in filters:
        
        phot_bands.append('2MASSKs')
        
        try:
            
            phot_refs.append(ref_dict['2MKS'])
            
        except:
            
            phot_refs.append(ref_dict['2MR1KS'])
    
    #Near-infrared (Akari)
    if 'AKARI9' in filters:
        
        phot_bands.append('AKARI_IRC9')
        phot_refs.append(ref_dict['AKARI9'])
    
    if 'AKARI18' in filters:
        
        phot_bands.append('AKARI_IRC18')
        phot_refs.append(ref_dict['AKARI18'])
    
    #Near- and mid-infrared (WISE)
    if 'WISE3P4' in filters: 
        
        phot_bands.append('WISE_W1')
        phot_refs.append(ref_dict['WISE3P4'])
    
    if 'WISE4P5' in filters: 
        
        phot_bands.append('WISE_W2')
        phot_refs.append(ref_dict['WISE4P5'])
        
    if 'WISE12' in filters: 
        
        phot_bands.append('WISE_W3')
        phot_refs.append(ref_dict['WISE12'])
        
    if 'WISE22' in filters: 
        
        phot_bands.append('WISE_W4')
        phot_refs.append(ref_dict['WISE22'])
    
    #Mid-infrared (Spitzer MIPS)
    if 'MIPS24' in filters: 
        
        phot_bands.append('Spitzer_MIPS24')
        
        if target == 'HD32297': 
            
            phot_refs.append('2008ApJ...686L..25M')
                
        elif target == 'TWA7':
            
            phot_refs.append('2005ApJ...631.1170L')
        
        elif target == 'HD181327':
            
            phot_refs.append('2006ApJ...650..414S')
        
        elif target == 'HD73350' or target == 'HD197481':
            
            phot_refs.append('2009ApJ...698.1068P')
            
        elif target == 'HD142091':
            
            phot_refs.append('2022MNRAS.517.2546L')
        
        elif target in beichman_06:
            
            phot_refs.append('2006ApJ...652.1674B')
        
        elif target in trilling_07:
            
            phot_refs.append('2007ApJ...658.1289T')
            
        elif target in trilling_08:
            
            phot_refs.append('2008ApJ...674.1086T')

        elif target in bryden_09:
            
            phot_refs.append('2009ApJ...705.1226B')
        
        elif target in morales_09:
            
            phot_refs.append('2009ApJ...699.1067M')
        
        elif target in carpenter_09:
            
            phot_refs.append('2009ApJ...705.1646C')
        
        elif target in kospal_09:
            
            phot_refs.append('2009ApJ...700L..73K')
        
        elif target in su_06:
            
            phot_refs.append('2006ApJ...653..675S')
        
        elif target == 'HD165908' or target == 'HD23356' or target == 'HD131511' or target == 'HD202628':
            
            phot_refs.append('2010ApJ...710L..26K')
        
        elif target in chen_14_data['Name'] :
            
            argval = np.where(chen_14_data['Name'] == target)
            
            if chen_14_data['r_F24'][argval].mask != True:
                
                ref = int(chen_14_data['r_F24'][argval].data[0])
                refind = np.where(chen_14_refs['Ref'] == ref)
                phot_refs.append(chen_14_refs['BibCode'][refind][0])
            
        else: 
            
            phot_refs.append('IPAC_IRSA_SEIPC')
        
    if 'MIPS70' in filters: 
        
        phot_bands.append('Spitzer_MIPS70')
        
        if target == 'HD32297': 
            
            phot_refs.append('2008ApJ...686L..25M')
        
        elif target == 'TWA7':
            
            phot_refs.append('2005ApJ...631.1170L')
        
        elif target == 'HD73350' or target == 'HD197481' or target == 'TYC93404371':
            
            phot_refs.append('2009ApJ...698.1068P')
        
        elif target == 'HD37594':
            
            phot_refs.append('PhillipsN.2011.PhDT')
        
        elif target == 'HD165908' or target == 'HD23356' or target == 'HD131511' or target == 'HD202628':
            
            phot_refs.append('2010ApJ...710L..26K')
        
        elif target == 'HD142091':
            
            phot_refs.append('2022MNRAS.517.2546L')
        
        elif target in kospal_09:
            
            phot_refs.append('2009ApJ...700L..73K')
        
        elif target in su_06:
            
            phot_refs.append('2006ApJ...653..675S')
        
        elif target in chen_14_data['Name'] :
            
            argval = np.where(chen_14_data['Name'] == target)
            
            if chen_14_data['r_F70'][argval].mask == False:
                
                ref = int(chen_14_data['r_F70'][argval].data[0])
                refind = np.where(chen_14_refs['Ref'] == ref)
                phot_refs.append(chen_14_refs['BibCode'][refind][0])
            
        else: 
            
            phot_refs.append('UNDEFINED')
    
    #Far-infrared (Herschel)
    if 'PACS70' in filters: 
        
        phot_bands.append('Herschel_PACS70')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['PACS70'])
    
    if 'PACS100' in filters: 
        
        phot_bands.append('Herschel_PACS100')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['PACS70'])
    
    if 'PACS160' in filters: 
        
        phot_bands.append('Herschel_PACS160')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['PACS70'])
    
    #Sub-millimetre (Herschel)
    if 'SPIRE250' in filters: 
        
        phot_bands.append('Herschel_SPIRE250')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['SPIRE250'])
    
    if 'SPIRE350' in filters: 
        
        phot_bands.append('Herschel_SPIRE350')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['SPIRE350'])
    
    if 'SPIRE500' in filters: 
        
        phot_bands.append('Herschel_SPIRE500')
        
        if target in dunes:
            
            phot_refs.append('2013A&A...555A..11E')
        
        elif target in debris_a:
            
            phot_refs.append('2014MNRAS.445.2558T')
        
        elif target in debris_fgk: 
            
            phot_refs.append('2018MNRAS.475.3046S')
        
        elif target == 'GJ581':
            
            phot_refs.append('2012A&A...548A..86L')
        
        elif target in morales_16:
            
            phot_refs.append('2016ApJ...831...97M')
        
        elif target in marshall_21:
            
            phot_refs.append('2021MNRAS.501.6168M')
        
        else:
            phot_refs.append(ref_dict['SPIRE500'])
    
    #Sub-millimetre (JCMT/SCUBA-2)
    if 'WAV450' in filters:
        
        phot_bands.append('JCMT_SCUBA2_450')
        
        if target in sons:
            
            phot_refs.append(ref_dict['WAV450'])
            
        else:
            phot_refs.append('UNDEFINED')
    
    if 'WAV850' in filters:
        
        phot_bands.append('JCMT_SCUBA2_850')
        
        if target == 'HD39060':
            
            phot_refs.append('')
        
        elif target == '':
            
            phot_refs.append('')
        
        elif target in sons:
            
            phot_refs.append(ref_dict['WAV850'])
        
        else:
            phot_refs.append('UNDEFINED')
    
    #Sub-millimetre (SMA) and Millimetre (ALMA)
    if 'WAV870' in filters or 'WAV1200' in filters or 'WAV1300' in filters or 'WAV1340' in filters:
        
        phot_bands.append('ALMAorSMA')
        
        if target == 'HD16743':
            
            phot_refs.append('2023MNRAS.521.5940M')
        
        elif target == 'HD138965': 
            
            phot_refs.append('2025MNRAS.541...71M')
        
        elif target == 'HD142091':
            
            phot_refs.append('2022MNRAS.517.2546L')
        
        elif target in reasons:
            
            phot_refs.append('2025A&A...693A.151M')
            
        elif target in steele_16:
            
            phot_refs.append('2016ApJ...816...27S')
            
        else:
            
            phot_refs.append('UNDEFINED')
    
    # print('%'*12)
    
    # print(target)
    
    # for j in range(0,len(phot_bands)):
        
    #     print(phot_bands[j],phot_refs[j])
    
    # print('%'*12)
    
    zipped_lists = dict(zip(phot_bands,phot_refs))
    photometry_references[target] = zipped_lists
    
    if 'UNDEFINED' in phot_refs or '--' in phot_refs:
        print(target, ' undefined photometry band(s)')
    
    with open(direc+'photometry_refences.json', 'w') as fp:
        json.dump(photometry_references,fp)