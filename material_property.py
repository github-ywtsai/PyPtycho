import numpy as np
import pandas as pd
import os

## These functions are created by 李柏緯

def energy_eV_to_wavelength_m(energy = None):
    h=6.62607015*10**-34
    e=1.60217651019*10**-19
    c=299792458.0
    wavelength=(h*c)/(energy*e)
    
    return wavelength


class atomic_database:
    def __init__(self,arg_atomic_name):
        self.initialize(arg_atomic_name)
        
        
    def initialize(self,arg_atomic_name):
        # the periodic table is downloaded form https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-mass/
        ## load from periodic table
        periodic_table_fp = os.path.join('.','database','periodic_table.csv')
        periodic_table_df = pd.read_csv(periodic_table_fp)
        df = periodic_table_df[periodic_table_df['Symbol'] == arg_atomic_name]
        if df.empty:
            print('Atomic name cannot be matched in database.')
            return
        self.atomic_number = df['AtomicNumber'].iloc[0]
        self.symbol = df['Symbol'].iloc[0]
        self.name = df['Name'].iloc[0]
        self.atomic_mass = df['AtomicMass'].iloc[0]
        self.atomic_radius = df['AtomicRadius'].iloc[0]
        self.ionization_energy = df['IonizationEnergy'].iloc[0]
        self.melting_point = df['MeltingPoint'].iloc[0]
        self.boiling_point = df['BoilingPoint'].iloc[0]
        self.density = df['Density'].iloc[0]
        
        # the table of f0 vs. sin(th)/lambda is copied from "International Tables for Crystallography: Volume C_ Mathematical, Physical and Chemical Tables" table 6.1.1.1 on page 555
        # the f1 and f2 are copied from Nist searching table
        # the f1 on NIST is equal to the f0 + f1 (at theta = 0) in this code 
        
        ## load f0 table
        # the first column in f0_talbe.csv is the sin(th)/lambda and then the f0 for all elements.
        # theta in rad and lambda in Angstrom
        class f0_table:
            def __init__(self):
                self.sin_th_over_lambda = None
                self.value = None
        self.f0_table = f0_table()
        f0_table_fp = os.path.join('.','database','f0_table.csv')
        f0_table_df = pd.read_csv(f0_table_fp)
        sin_th_over_lambda = f0_table_df.iloc[:,0].values
        value = f0_table_df.iloc[:,self.atomic_number].values
        self.f0_table.sin_th_over_lambda = sin_th_over_lambda
        self.f0_table.value = value
        
        ## load f1 and f2  table
        # in f1f2talbe, 3 cols are a set, where the 1st col is energy (in keV), 2nd col is f1, and 3rd col is f2
        class f1_table:
            def __init__(self):
                self.energy = None # in eV
                self.value = None
        class f2_table:
            def __init__(self):
                self.energy = None # in eV
                self.value = None  
        self.f1_table = f1_table()
        self.f2_table = f2_table()
        f1f2_table_fp = os.path.join('.','database','f1f2_table.csv')
        f1f2_table_df = pd.read_csv(f1f2_table_fp,header=None)
        self.f1_table.energy = f1f2_table_df.iloc[:,self.atomic_number*3-3].values*1E3 # convert keV to eV
        self.f1_table.value  = f1f2_table_df.iloc[:,self.atomic_number*3-3+1].values - self.f0_table.value[0]
        self.f2_table.energy = f1f2_table_df.iloc[:,self.atomic_number*3-3].values*1E3 # convert keV to eV
        self.f2_table.value  = f1f2_table_df.iloc[:,self.atomic_number*3-3+2].values
    
    def calculate_f0(self, energy=None, theta=None):
        # energy: the photon energy of X-ray in eV
        # theta: the incident angle (in degree when input and rad in code)

        if energy is None:
            energy = 9000
            print('Default energy is 9000 eV.')
        if theta is None:
            theta = 0
            print('Default theta is 0 degree.')
            
        theta = np.deg2rad(theta)
       
        wavelength= energy_eV_to_wavelength_m(energy) * 1E10 ### 1E3: keV to eV, 1E10: m to angstrom
        sin_th_over_lambda=np.sin(theta)/wavelength
        f0_interp = np.interp(sin_th_over_lambda,self.f0_table.sin_th_over_lambda,self.f0_table.value)
        
        return f0_interp
        


    def calculate_f1(self, energy=None):
        # energy: the photon energy of X-ray in eV
        if energy is None:
            energy = 9000
            print('Default energy is 9000 eV.')      
        
        f1_interp = np.interp(energy,self.f1_table.energy,self.f1_table.value)

        return f1_interp

    def calculate_f2(self, energy=None):
        # energy: the photon energy of X-ray in eV
        if energy is None:
            energy = 9000
            print('Default energy is 9000 eV.')      
        
        f2_interp = np.interp(energy,self.f2_table.energy,self.f2_table.value)

        return f2_interp

    
    def calculate_refractive_index(self,energy=None, theta=None):
    # example: calculating Refractive index for gold Au at 9.030794 keV
    # n(wavelength) = 1 - delta(wavelength) + i beta(wavelength)
    #f1 = 74.9382 # [e/atom] from NIST, equal to f0(theta=0) + f1 in code
    #f2 = 6.0726 # [e/atom] % from NIST
    #re = 2.8179403e-15 # classical electron rauius [m]
    #Na = 6.02214129e23 # Avogadro constant [1/mol]
    #Ma = 196.966569 # molar mass [g/mol]
    #rho = 19.32 # density [g/cm^3]
    #rho = rho / (1e-2)**3 # convert unit from [g cm-3] to [g m-3]
    #na = rho*Na/Ma # number density+
    #delta = na*re*wavelength**2/2/pi*f1
    #beta =  na*re*wavelength**2/2/pi*f2
    #n_refractivity = 1 - delta + 1j * beta
    
        # energy input in eV
        if energy is None:
            energy = 9000
            print('Default energy is 9000 eV.')
        if theta is None:
            theta = 0
            print('Default theta is 0 degree.')
        
        re = 2.8179403e-15 # classical electron rauius [m]
        Na = 6.02214129e23 # Avogadro constant [1/mol]
        Ma = self.atomic_mass # molar mass [g/mol] # atomic mass and molar mass are equal in amount but different in unit
        rho = self.density # density [g/cm^3]
        rho = rho / (1e-2)**3 # convert unit from [g cm-3] to [g m-3]
        na = rho*Na/Ma # number density
    
        wavelength= energy_eV_to_wavelength_m(energy)
        f0 = self.calculate_f0(energy = energy, theta = theta)
        f1 = self.calculate_f1(energy=energy)
        f2 = self.calculate_f2(energy=energy)
    
        delta = na*re*wavelength**2/2/np.pi*(f0+f1)
        beta =  na*re*wavelength**2/2/np.pi*f2
        refractive_index = 1 - delta + 1j * beta
        
        return refractive_index