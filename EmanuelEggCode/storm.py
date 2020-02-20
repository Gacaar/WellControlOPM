#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Basic code for MNPO and other simulations for the STORMS research group.
"""
from __future__ import absolute_import
from ecl.summary import EclSum
from backend import StormEclSupport
import glob
import inspect
import numpy as np
import subprocess #subprocess.call(arglist)
import os
import scipy.io as sio
import matplotlib.pyplot as plt

# TODO : Eventual

class StormBackend:
	"""
		Class for STORM Backend operations (MNPO, simulations)

		Args:
			caseName (str): Model's name (EGG, OLYMPUS) to be used in all file names.

		Attributes:
			costs (dict): A dictionary with oil, water and gas production/injection costs.
			caseName (str): The model name, to be used in all file names.
			dumpName (str): A folder that will receive the temporary files to avoid cluttering.
			iterFname (str): A base name for all iteration files.
			verboseFlags (tuple): Sets the solver and time step verbosities.
			baseFolder (str): Folder containing the base model for simulation.
			wells (dict): Wells information (can be recovered from a summary file as well).
			numWells (int): Number of wells.
			phases (tuple): Oil, Water and Gas Phases (enabled by default)
			curData (EclSum): Points to a summary file.
			rates (dict): All rates info.
			currentControl (list): Current production control array for producers.
			currentInjectionControl (list): Current injection control array for injectors.
			currentDelta (list): Current delta for control.
			algorithmSpan (int): Number of control possibilities per iteration.
			deltaParams (dict): Delta equation parameters.
			producerSat (list): Producers' oil saturation.
			curIteration (int): Holds the current iteration.
			maxIterations (int): Maximum number of iterations (default: 40).
			dt (float): Time step length (in days).
			t0 (float): Simulation's initial time (days, default: 0).
			eps (float): Numerical correction factor for Modified NPV (MNPV).
			s0 (numpy array): Initial well saturations in a given iteration.
			optWellNPVs (list): Best Well NPVs.
			optWellMNPVs (list): Best Well MNPVs.
			optWellBHPs (list): Best Well BHPs.
			optWellSat (list): Best Well Oil Saturations.
			optInjections (list): Best Well Injection Rates.
			iterWellNPVs (list): Iteration Well NPVs.
			iterWellMNPVs (list): Iteration Well MNPVs.
			iterWellBHPs (list): Iteration Well BHPs.
			iterWellSat (list): Iteration Well Oil Saturations.
			iterWellOilRates (list): Iteration Well Oil Rates.
			iterWellWaterRates (list): Iteration Well Water Rates.
			iterDelta (list): Iteration Deltas.
	"""
	def __init__(self,caseName,**kwargs):
		"""Class constructor.
		
		Args:
		caseName (str): Model's name (EGG, OLYMPUS) to be used in all file names.	

		Key Args:
			costs (dict): Costs information. Should have the following keywords:
				'OilProd','WaterProd','GasProd','OilInj','WaterInj','GasInj'
			producers (list): Producers' names
			injectors (list): Injectors' names
			prodControl (list): Producers' initial control
			injeControl (list): Injectors' initial control
			span (int): Possibility tree's span for MNPO (default: 9)
		"""
		self.costs = kwargs['costs'] 		# Cost initialization
		self.caseName = caseName			# Case Name
		self.dumpName = 'DUMP'				# Dump folder (TODO: make this flexible)
		self.iterFname = 'ITER{}I{}'		# Iterations base name
		self.verboseFlags = (0,0)			# Verbose Flags (TODO: change this)
		self.baseFolder = 'BASE'			# Folder with the base .DATA file
		prodNames = kwargs['producers']		# Producer Wells (names)
		injNames = kwargs['injectors']		# Injector wells (names)

		self.wells = {						#
			'prod' : prodNames,				# Wells dictionary
			'inje' : injNames				#
		}									#

		self.numWells = len(self.wells['prod'] + self.wells['inje']) # Get number of wells
		self.prodConstraints = kwargs['prodConstraints']
		self.injeConstraints = kwargs['injeConstraints']

		#TODO: nao tem gas no EGG MODEL
		self.phases = (1,1,0) 				# Phases: oil, water, gas
		self.curData = None					# Pointer to a summary file 			

		self.rates = {						# Variable which holds rates info.
			'wopr' : [],
			'wwir' : [],
			'wwpr' : [],
			'wwir' : [],
			'wgpr' : [],
			'wgir' : []
		}

		_p, _i = len(self.wells['prod']), len(self.wells['inje'])		# Number of producers and number of injectors

		self.currentControl = kwargs['prodControl']						#
		self.currentInjectionControl = kwargs['injeControl']			# Initial BHPs, rates and delta
		self.currentDelta = np.array([0. for i in range(_p)])			#

		self.algorithmSpan = 9 if 'span' not in kwargs.keys() else kwargs['span']	# Number of possibilities (span)

		self.deltaParams = {											# Delta parameters (TODO: make this flexible)
			'k' : np.array([5 for i in range(_p)]), 
			'kappa' : np.array([1e-3 for i in range(_p)]), 
			'h' : np.array([1e-2 for i in range(_p)])
		}

		self.producerSat = np.array([1. for i in range(_p)])			# Initial producers' oil saturation

		self.__deltaEquation(np.zeros(_p))								# Initial delta calculation (private method)

		self.curIteration = 0		# Current Iteration
		self.maxIterations = 80 if 'maxIterations' not in kwargs.keys() else kwargs['maxIterations']	# Maximum number of iterations
		self.dt = 90 if 'timeStep' not in kwargs.keys() else kwargs['timeStep']							# Time step Length (TODO: flexible (all of these))
		self.eps = .04 if 'eps' not in kwargs.keys() else kwargs['eps']									# Epsilon for numerical correction in MNPV function
		self.t0 = 0	if 't0' not in kwargs.keys() else kwargs['t0']										# Initial time

		self.s0 = np.array([0. for i in range(_i)] + [1. for i in range(_p)]) # All initial well saturations

	def setVerbosity(self,solver,tstep):
		"""Sets verbosities (solver, terminal output) of OPM Flow.

		Args:
			solver (int): Solver verbosity
			tstep (int): Time step verbosity

		Returns:
			None
		"""
		self.verboseFlags = (solver,tstep)

	def __deltaEquation(self,ds):
		"""Calculates the delta parameters to be used in the MNPO, per production well.

		Args:
			ds (numpy array): Array of oil producers' saturation.

		Returns:
			None
		"""
		k, K, h = self.deltaParams['k'], self.deltaParams['kappa'], self.deltaParams['h'] # Unpacks delta parameters
		self.currentDelta = k * np.maximum(K, h * np.array(ds))					

	def __moveIters(self):
		"""Moves the iteration files to the dump folder.

		Args:
			None.

		Returns:
			None.
		"""
		iterFiles = [files for files in glob.glob('*.DATA')]
		iterFiles.extend([files for files in glob.glob('*.INC') if 'SCH' in files])		# Gets all the iteration case files and their correspondent schedules.
		for files in iterFiles:
			subprocess.call(['mv',files,'{}'.format(self.dumpName)])		# Move the files to the dump folder.

	def execFlow(self,fname,output='true'):
		"""Executes OPM flow.

		Args:
			fname (str): File to be executed.
			output (str='true'): Enables/Disables OPM's terminal output.

		Returns:
			None.
		"""
		_exec =	[	# Generates commands for OPM Flow
			'flow',fname,'--output-dir={}'.format(self.dumpName),
			'--solver-verbosity={}'.format(self.verboseFlags[0]),
			'--time-step-verbosity={}'.format(self.verboseFlags[1]),
			'--enable-opm-rst-file=true','--enable-terminal-output={}'.format(output),
		]
		subprocess.call(_exec)	# Call OPM Flow



	def clearDump(self):
		"""Clears the dump folder.

		Args:
			None.

		Returns:
			None.
		"""
		for i in glob.glob('{}/*.*'.format(self.dumpName)): os.remove(i)

	def genVectors(self,well):
		"""Generates rate vectors from the summary file currently loaded in the class.
			
		Args:
			well (str): Well to get the data from.

		Raises:
			StormBackendException: Summary file not found or loaded

		Returns:
			wopr (numpy array): Well Oil Production Rate
			wgpr (numpy array): Well Gas Production Rate
			wwpr (numpy array): Well Water Production Rate
			woir (numpy array): Well Oil Injection Rate
			wgir (numpy array): Well Gas Injection Rate
			wwir (numpy array): Well Water Injection Rate
		"""
		try:
			_n = len(self.curData.get_days())	# Get number of elements
		except:
			raise StormBackendException('Summary file not found or loaded.')

		if self.phases[0] != 0:
			try: wopr = self.curData.numpy_vector('WOPR:{}'.format(well))	# Gets oil production rates, or zeros if they do not exist
			except: wopr = np.zeros(_n)
			try: woir = self.curData.numpy_vector('WOIR:{}'.format(well))	# Gets oil injection rates, or zeros if they do not exist
			except: woir = np.zeros(_n)
		else:
			wopr,woir = np.zeros(_n),np.zeros(_n)

		if self.phases[1] != 0:
			try: wwpr = self.curData.numpy_vector('WWPR:{}'.format(well))	# Gets water production rates, or zeros if they do not exist
			except: wwpr = np.zeros(_n)
			try: wwir = self.curData.numpy_vector('WWIR:{}'.format(well))	# Gets oil injection rates, or zeros if they do not exist
			except: wwir = np.zeros(_n)
		else:
			wwpr,wwir = np.zeros(_n),np.zeros(_n)

		if self.phases[2] != 0:
			try: wgpr = self.curData.numpy_vector('WGPR:{}'.format(well))	# Gets gas injection rates, or zeros if they do not exist
			except: wgpr = np.zeros(_n)
			try: wgir = self.curData.numpy_vector('WGIR:{}'.format(well))	# Gets gas injection rates, or zeros if they do not exist
			except: wgir = np.zeros(_n)
		else:
			wgpr,wgir = np.zeros(_n),np.zeros(_n)

		return wopr,woir,wwpr,wwir,wgpr,wgir

	def getBHPs(self,well):
		"""Gets a well's Bottom Hole Pressure (BHP)

		Args:
			well (str): Well to get BHP from.

		Raises:
			StormBackendException: BHP data not found.

		Returns:
			wbhp (numpy array): Well BHP
		"""
		try: wbhp = self.curData.numpy_vector('WBHP:{}'.format(well))
		except: raise StormBackendException('BHP data not found.')
		return wbhp

	def getSaturation(self,well):
		"""Calculates the oil saturation for a well.

		Args:
			well (str): Well to get the oil saturation from.

		Raises:
			StormBackendException: Summary file not found or loaded

		Returns:
			satOil (numpy array): Well's oil saturation
		"""
		_n = len(self.curData.get_days())
		wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)
		fpr = wopr+wgpr+wwpr
		satOil = np.zeros(_n)
		for i in range(_n):
			if fpr[i] != 0: satOil[i] = wopr[i]/fpr[i]
		return satOil

	def __initMNPOVars(self):
		"""Initalializes the variables which will hold MNPO's various data.
	
		Args:
			None.

		Returns:
			None.
		"""
		self.optWellNPVs = []			# Best Well NPVs
		self.optWellMNPVs = []			# Best Well MNPVs
		self.optWellBHPs = []			# Best Well BHPs
		self.optWellSat = []			# Best Well Oil Saturations
		self.optInjections = []			# Best Well Injections
		self.iterWellMNPVs = []			# Iteration Well MNPVs
		self.iterWellBHPs = []			# Iteration Well BHPs
		self.iterWellSat = []			# Iteration Well Oil Saturations
		self.iterWellOilRates = []		# Iteration Well Oil Rates
		self.iterWellWaterRates = []	# Iteration Well Water Rates
		self.iterDelta = []				# Iteration Deltas
		
		self.rates = {					# Rates initialization for MNPO
			'wopr' : [[] for i in range(self.numWells)],
			'wwir' : [[] for i in range(self.numWells)],
			'wwpr' : [[] for i in range(self.numWells)],
			'wwir' : [[] for i in range(self.numWells)],
			'wgpr' : [[] for i in range(self.numWells)],
			'wgir' : [[] for i in range(self.numWells)]
		}

	def loadSummary(self,fname):
		"""Loads a summary file to the @curData attribute.

		Args:
			fname (str): Summary File Name

		Raises:
			StormBackendException: File not found.

		Returns:
			Nones.
		"""
		try: self.curData = EclSum(fname)	# Creates an Eclipse Summary class for data handling
		except: raise StormBackendException('Eclipse Summary File "{}" not found.'.format(fname))

	def wellIndex(self,well):
		"""Returns a well index (a number which represents a well), akin to a h

		Args:
			well (str): Well to be indexed.

		Raises:
			StormBackendException: Well not found.

		Returns:
			int: Well "hash" (index).
		"""
		wells = self.wells['inje'] + self.wells['prod']
		try: return wells.index(well)
		except ValueError:
			raise StormBackendException('Well "{}" not found in the model'.format(well))

	def calculateNPV(self,**kwargs):
		"""Calculates NPV given a summary file previously loaded in the class.
		The NPV can be either the standard or the Modified NPV (MNPV).

		Args:
			None.

		Key Args:
			mnpv (bool): Flag indicating if modified NPV (default: False)
			eps (double): Numerical corrector flag (default: 0.01)
			t0 (double): Initial time in days (default: 0)
		Raises:
			StormBackendException: Summary file not found or loaded.

		Returns:
			dict: A dictionary containing the following fields:
				wellNPVs  - Well NPVs
				NPV       - Sum of well NPVs
				oilProd   - Oil Rates
				waterProd - Water Rates
				waterInj  - Water Injections
				t0        - Initial Time 
				s0        - Initial Saturations
				t         - Time Vector
				s         - Saturation Vector
				modified  - Flag that indicates if the NPV is modified (MNPV) or NPV.
		"""
		modified = False if 'mnpv' not in kwargs.keys() else kwargs['mnpv']	#
		_eps = .01 if 'eps' not in kwargs.keys() else kwargs['eps']			# Variable initializations.
		eps = _eps / (1. - _eps)											#
		t0 = 0 if 't0' not in kwargs.keys() else kwargs['t0'] 				#

		try:
			wells = list(self.curData.wells())								# Get well info
			t = self.curData.get_days()										# Time info
		except:
			raise StormBackendException('Summary file not found or loaded.')

		_n = len(self.curData.get_days())									# Array size for time and rate handling
		_w = len(wells)														# Auxiliar for well number
		s0 = np.ones(_w) if 's0' not in kwargs.keys() else kwargs['s0']		# Initial saturations (MNPV case)
		_dim = (_w,_n)														# Dimensions

		oilRevenue, waterCost, injectionCost = np.zeros(_dim), np.zeros(_dim), np.zeros(_dim) # Array initializations
		sats = np.zeros(_dim)												# Array initializations

		cO = 0.0 if 'OilProd' not in self.costs.keys() else self.costs['OilProd'] 		#
		cW = 0.0 if 'WaterProd' not in self.costs.keys() else self.costs['WaterProd'] 	#
		cG = 0.0 if 'GasProd' not in self.costs.keys() else self.costs['GasProd'] 		# Cost Unpacking
		cOi = 0.0 if 'OilInj' not in self.costs.keys() else self.costs['OilInj'] 		#
		cWi = 0.0 if 'WaterInj' not in self.costs.keys() else self.costs['WaterInj'] 	#
		cGi = 0.0 if 'GasInj' not in self.costs.keys() else self.costs['GasInj']		#

		for well in wells:																#
			wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)						# Get rates for each well
			wIdx = self.wellIndex(well)													# Index current well
			s = self.getSaturation(well)												# 
			sats[wIdx,:] = s 															# Get saturation for current well
			for i in range(_n):
				dT = t[i] - t0 if i == 0 else t[i] - t[i-1]								# Calculate dt for NPV
				dS = abs(s[i] - s0[wIdx]) if i == 0 else abs(s[i] - s[i-1])				# Calculate ds for MNPV
				if s0[wIdx] == 0 and i == 0: dS = 0
				_den = dS + eps if modified else 1.										# Denominator for MNPV
				oilRevenue[wIdx,i] = dT*cO*wopr[i]/_den 								#
				waterCost[wIdx,i] = dT*cW*wwpr[i]/_den 									# NPV Revenue and Costs calculation
				injectionCost[wIdx,i] = dT*cWi*wwir[i]/_den  							#

		npvMatrix = oilRevenue - (waterCost + injectionCost)							# Final Calculation (NPV)
		npvPeriod = [np.sum(npvMatrix[:,i]) for i in range(_n)]							#
		_dic = {																		# Bundle results
			'wellNPVs' : npvMatrix, 'NPV' : npvPeriod,
			'oilProd' : oilRevenue, 'waterProd' : waterCost,
			't0' : t0, 's0' : s0, 'modified' : modified,
			'waterInj' : injectionCost, 't' : t, 's' : sats
		}

		return _dic

	#AQUI EH ONDE ACONTECE A BAGACEIRA
	def MNPOAlgorithm(self,verbosities=(True,'false')):
		"""MNPO's Main Algorithm Phase. Executes MNPO until a negative NPV is found.

		Args:
			verbosities (tuple); controls STORMS and OPM verbosities. Default: (True, 'false').

		Returns:
			None.
			
		"""
		def deltaS():
			"""An inner function to compute ds for delta calculation.

			Args:
				None.

			Returns:
				ds (numpy array): Saturation Variation for Delta Equation.

			"""
			s = self.optWellSat[-1].copy()
			k = 0
			ds = np.zeros(len(self.wells['prod']))		# ds initialization
			for well in self.wells['prod']:				# ds for each producer well
				j = self.wellIndex(well)
				ds[k] = abs(s[j] - self.s0[j])			# ds = |s - s0|
				k += 1
			print(ds)
			return ds
			
		self.__initMNPOVars()											# Init MNPO variables
		self.curIteration = 0											# Current iteration: 0

		# Initialize injection control variables
		# forgetting factor
		forg_factor = 0.4

		# initial values
		theta_w = np.array([1.0, 0.0])
		P_w = 1000.0 * np.eye(2) 
		K_w = np.array([0.0, 0.0]) 

		theta_l = np.array([1.0, 0.0])
		P_l = 1000.0 * np.eye(2) 
		K_l = np.array([0.0, 0.0]) 
		_p, _i = len(self.wells['prod']), len(self.wells['inje'])		# Number of producers and number of injectors
		q_hat_w, q_hat_l = 0.,0.
		fwpr_max, flpr_max = 1600., 14000.
		while self.curIteration < self.maxIterations:					# Run until maxIterations



			### INJECTION CONTROL
			if self.curIteration > 1:
			
				
				inj, theta_w, K_w, P_w, q_hat_w, theta_l, K_l, P_l, q_hat_l = \
					calc_inj(self.curData, old_data,  theta_w, P_w, K_w, q_hat_w, theta_l, P_l, K_l, q_hat_l, forg_factor, fwpr_max, flpr_max, self.curIteration)
	
				
				self.currentInjectionControl  = [inj/_i for i in range(_i)]

			########################

			print('Iteration {}'.format(self.curIteration+1))
			if verbosities[0]:											# Verbosity check
				print('Current Delta: {}'.format(self.currentDelta))
				print('s0: {}'.format(self.s0))
				print('t0: {} days'.format(self.t0))
			positive = self.MNPOIteration(verbosities)					# Check if a negative NPV is yielded
			if positive:												# If NPV is positive, update info for next iteration
				self.__deltaEquation(deltaS())							# Delta Update
				self.s0 = self.optWellSat[-1].copy()					# s0 Update
				self.t0 += self.dt 										# t0 Update
				self.currentControl = self.optWellBHPs[-1].copy()		# Control Update
				self.curIteration += 1									# Iteration Update
			else:
				break													# Negative NPV: break this loop.

			old_data = self.curData

		self.__moveIters()


	# ESSE EH O METODO EXECUTADO EM CADA ITERACAO!
	def MNPOIteration(self,verbosities=(True,'false')):
		"""Runs a MNPO Iteration. Called by the MNPOAlgorithm() method.

		Args:
			verbosities (tuple); controls STORMS and OPM verbosities. Default: (True, 'false').

		Returns:
			positive (bool): A flag indicating if the resultant NPV is positive
		"""
		verbose = verbosities[0]												# STORMS Verbosity setting
		iteration = self.curIteration											# Current iteration
		fIn = '{}/{}.DATA'.format(self.baseFolder,self.caseName)				#
		fIterEnd = '{}{}.DATA'.format(iteration,self.caseName)					# File base initializations
		rstName = '{}/{}{}'.format(self.dumpName,iteration-1,self.caseName)		#
		scheduleEnd = 'SCHSTEP_{}.INC'.format(iteration)						#

		config = {																# Configurations for file building
			'restart' : False,
			'schName' : scheduleEnd
		}
		configIter = {															# Configurations for iteration file building
			'restart' : False
		}

		if iteration != 0:
			config['restart'], configIter['restart'] = True, True 				#
			config['rstName'], configIter['rstName'] = rstName, rstName			# Configurations for restart cases
			config['rstIndex'], configIter['rstIndex'] = iteration, iteration 	#

		_n = self.algorithmSpan													# _n: Local variable for algorithm span

		_window1 = np.array([self.currentControl[:] for i in range(_n)])		# first window controls
		_window2 = np.array([self.currentControl[:] for i in range(_n)])		# second window controls
		_w = len(self.wells['prod'])											# _w: number of producers
		_dv = int(_n/2)															# _dv: extreme point for delta span after first iteration
		if iteration == 0:														#
			dmul = [i for i in range(0,-_n,-1)]									# delta span for first iteration
		else:																	#
			dmul = [i for i in range(-_dv,_dv+1)]								# delta span for next iterations

		self.iterDelta.append(self.currentDelta)								# Save delta to iter results
		maxMNPVWell = [(-1e20, 0) for i in range(_w)]							# Initialize max MNPV for each well with low values and well index 0
		maxTempMNPVWell = maxMNPVWell[:]										# Copy above variable
		maxSats = [1. for i in range(_w)]										# Saturation for maximum MNPV
		maxWopr = [0. for i in range(_w)]										# Oil Rate for maximum MNPV
		maxWwpr = [0. for i in range(_w)]										# Water Rate for maximum MNPV

		bestCtrl = np.array(self.currentControl)								# Array to hold the best control

		mnpvParams = {															# mnpvParams: config for calcNPV(), MNPV case
			't0' : self.t0,
			's0' : self.s0,
			'mnpv' : True,
			'eps' : self.eps
		}
		npvParams = {															# mnpvParams: config for calcNPV(), NPV case
			't0' : self.t0,
			'mnpv' : False
		}
		infoLst = []															# Iteration info
		# Aqui eh onde o controle acontece de fato!
		# PAREI POR AQUI!

		# TODO: 
		# Ver nos artigos e no mestrado do Bulba se tem alguma explicação envolvendo testar várias possibilidades (span of possibilities) e também sobre janelas de controle
		# Tambem aproveitar e procurar alguma coisa sobre delta


		for i in range(_n):																									# For each possibility in span...
			_window1[i,:] = np.minimum(self.prodConstraints[1], \
							np.maximum(self.prodConstraints[0], _window1[i,:] * (np.ones(_w) + dmul[i]*self.currentDelta)))		# Gen Window 1 BHPs
			_window2[i,:] = np.minimum(self.prodConstraints[1], \
							np.maximum(self.prodConstraints[0], _window2[i,:] * (np.ones(_w) + 2.*dmul[i]*self.currentDelta)))	# Gen Window 2 BHPs
			if verbose:																										#
				print('\tSpan {}:'.format(iteration, i+1))																	# Control verbosity
				print('\tControls:\n\t -{}\n\t -{}'.format(_window1[i,:],_window2[i,:]))									#

			iterFile = self.iterFname.format(self.curIteration,i+1) + '.DATA'			# Prepare iter files
			configIter['schName'] = 'SCHITER{}_{}.INC'.format(self.curIteration,i+1)	#

			controlSch = {																# Prepare iter schedules
				'inje' : [self.currentInjectionControl, self.currentInjectionControl],
				'prod' : [_window1[i,:], _window2[i,:]]
			}
			# APLICANDO O CONTROLE DE FATO
			print("Preparando o arquivo de controle (schedule)")
			if self.curIteration > 1:
				StormEclSupport.schedule(configIter['schName'],'iter',self.wells,controlSch,self.dt,iteration,('BHP','RATE'))	# Storm Backend calls
			else:
				StormEclSupport.schedule(configIter['schName'],'iter',self.wells,controlSch,self.dt,iteration,('BHP','BHP'))	# Storm Backend calls
			# Talvez seja um ponto critico: criacao de arquivos
			print("Preparando os arquivos!")
			print("Base file name, fIn = " + fIn)
			print("Schedule file name = " + configIter['schName'])
			print("Output file name: iterFile = " + iterFile)
			StormEclSupport.configureEclFiles(fIn,iterFile,**configIter)													#
			# Imagino que seja aqui que da problema
			print("Executou ateh aqui! Ja deveria estar com tudo pronto para executar o FLOW e fazer a simulacao")
			self.execFlow(iterFile, verbosities[1])	
			print("Incrivel! Passou!")																		# Execute OPM Flow
			iterSummary = self.dumpName + '/' + self.iterFname.format(self.curIteration,i+1) + '.S0002'						#
			self.loadSummary(iterSummary)																					# Load corresponding summary

			infoLst.append(self.calculateNPV(**mnpvParams))																	# save MNPV info
			t = infoLst[i]['t']
			w1Idx = 0
			while t[w1Idx] - self.t0 <= self.dt and w1Idx < len(t): w1Idx += 1												# Get index for time before the second window
			k = 0
			for well in self.wells['prod']:																					# For each producer well...
				wopr, woir, wwpr, wwir, wgpr, wgir = self.genVectors(well)													# Get Rates
				j = self.wellIndex(well)
				mnpv = infoLst[i]['wellNPVs'][j]																			# Get current well's MNPV
				#sat = infoLst[i]['s'][j]
				#print(mnpv)
				if np.sum(mnpv) > maxMNPVWell[k][0]:																		# Update best MNPV info if it is better than 
					maxMNPVWell[k] = (np.sum(mnpv),i+1)																		# the previous MNPV:
					maxTempMNPVWell[k] = (np.sum(mnpv[:w1Idx]),i+1)															# Temp MNPV
					maxSats[k] = infoLst[i]['s'][j][w1Idx-1]																# Oil Saturation
					maxWopr[k] = np.sum(wopr[:w1Idx])																		# Oil Rate
					maxWwpr[k] = np.sum(wwpr[:w1Idx])																		# Water Rate
					bestCtrl[k] = _window1[i,k]																				# BHP Control
				k += 1
		self.iterWellMNPVs.append(infoLst)																					# Save MNPVs to output data
		if verbose:
			print('Best Control: {}'.format(bestCtrl))																		# More verbosity
		controlSch= {																										# Schedule for best Control
			'prod' : [bestCtrl],
			'inje' : [self.currentInjectionControl]
		}
		#print(type(controlSch['prod']))
		if self.curIteration > 1:
			StormEclSupport.schedule(config['schName'],'onestep',self.wells,controlSch,self.dt,iteration,('BHP','RATE'))		# Prepare files for best simulation
		else:
			StormEclSupport.schedule(config['schName'],'onestep',self.wells,controlSch,self.dt,iteration,('BHP','BHP'))			# Prepare files for best simulation
		StormEclSupport.configureEclFiles(fIn,fIterEnd,**config)															#
		self.execFlow(fIterEnd, verbosities[1])																				# Execute OPM Flow
		theSummary = '{}/{}{}.S{:04d}'.format(self.dumpName,iteration,self.caseName,iteration+1)							#
		self.loadSummary(theSummary)																						# Load corresponding summary file

		opNPVs = self.calculateNPV(**npvParams)																				# Calculate NPV for best control

		positive = np.sum(opNPVs['NPV']) > 0.																				# Check if the NPV is positive

		if positive:													# If NPV was positive...
			for well in self.curData.wells():		
				wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)	#
				wIdx = self.wellIndex(well)								# Save oil and water rates for each well
				self.rates['wopr'][wIdx].extend(wopr)					#
				self.rates['wwpr'][wIdx].extend(wwpr)					#
			self.optWellNPVs.append(opNPVs)								# Save best NPVs
			self.optWellMNPVs.append({									# Save best MNPVs (window 1, window 2, overall iteration)
				'w1' : maxTempMNPVWell,
				'w2' : maxMNPVWell,
				'it' : self.calculateNPV(**mnpvParams)
			})		
			self.optWellBHPs.append(bestCtrl)							# Save producers' control
			self.optInjections.append(self.currentInjectionControl)		# Save injectors' control
			self.optWellSat.append(opNPVs['s'][:,-1])					# Save oil saturation
			self.iterWellSat.append(maxSats)							# Save iter's well oil saturation
			self.iterWellOilRates.append(maxWopr)						# Save iter's well oil rates
			self.iterWellWaterRates.append(maxWwpr)						# Save iter's well water rates
		return positive													# Return flag indicating NPV > 0 (or not).

	def collectInfo(self):
		"""Given an execution of MNPO, collects the relevant info into a dictionary.

		Args:
			None.

		Returns:
			dict: A dictionary with the following keys:
				wellNPV: NPV for each well;
				wellMNPV: MNPV for each well;
				w1WellMNPVs: MNPV for each wells after the first window in predictor's phase;
				t: Time array;
				s: Oil saturation array;
				bhp: BHP values;
				NPV: Overall NPV array;
				MNPV: Overall MNPV array;
				cNPV: Cumulative NPV array;
				cMNPV: Cumulative MNPV array;
				w1WellSat: Well Oil Saturations after first window;
				w1WellOPR: Well Oil Rates after first window;
				w1WellWPR: Well Water Rates after first window;
				'wopr': Oil Production Rates;
				'wwpr': Water Production Rates;
				'wwir': Water Injection Rates;
				'injRates': Injection Control.

		"""
		_w = len(self.wells['prod'])
		npv, mnpv, t = [], [], []
		w1WellMNPVs = [[] for i in range(_w)]
		n = min([len(self.optWellNPVs),len(self.optWellMNPVs),len(self.optWellSat)])
		for i in range(n):
			_np = self.optWellNPVs[i]
			npv.extend(_np['NPV'])														# Appends NPV to the result array
			t.extend(_np['t'])															# Appends time data to time array
			if i == 0:
				wellNPVs = _np['wellNPVs'].copy()										# Appends info for well NPVs
			else:																		#
				wellNPVs = np.concatenate((wellNPVs,_np['wellNPVs']),axis=1)			#
			w1mnpv = self.optWellMNPVs[i]['w1']
			for k in range(_w):
				w1WellMNPVs[k].append(w1mnpv[k][0])										# MNPV info for first window

			_mnp = self.optWellMNPVs[i]['it']
			mnpv.extend(_mnp['NPV'])
			if i == 0:
				wellMNPVs = _mnp['wellNPVs'].copy()										# Appends info for well MNPVs
			else:																		#
				wellMNPVs = np.concatenate((wellNPVs,_mnp['wellNPVs']),axis=1)			#		
		results = {																		# Bundles the results into a dictionary for .mat file saving.
			'wellNPV' : wellNPVs,
			'wellMNPV' : wellMNPVs,
			't' : t,
			'w1WellMNPVs' : np.array(w1WellMNPVs),
			's' : np.array(self.optWellSat),
			'bhp' : np.array(self.optWellBHPs),
			'NPV' : npv,
			'MNPV' : mnpv,
			'cNPV' : np.cumsum(npv),
			'cMNPV' : np.cumsum(mnpv),
			'w1WellSat' : np.array(self.iterWellSat),
			'w1WellOPR' : np.array(self.iterWellOilRates),
			'w1WellWPR' : np.array(self.iterWellWaterRates),
			'wopr' : np.array(self.rates['wopr']),
			'wwpr' : np.array(self.rates['wwpr']),
			'wwir' : np.array(self.rates['wwir']),
			'injRates' : np.array(self.optInjections)
		}
		return results

	def MNPOPlots(self,iterations):
		"""Generates plots and .mat files with relevant info.

		Args:
			iterations (int): Number of iterations to plot data.

		Returns:
			None.
		"""
		i = 0																	#
		_n = [i for i in range(len(self.wells['prod'] + self.wells['inje']))]	#			
		s = 'DUMP/{0}EGG.S{1:04d}'												# (TODO: make this flexible)
		self.rates['wopr'] = [[] for i in _n]									#
		self.rates['woir'] = [[] for i in _n]									#
		self.rates['wwpr'] = [[] for i in _n]									# Initializations
		self.rates['wwir'] = [[] for i in _n]									#
		self.rates['wgpr'] = [[] for i in _n]									#
		self.rates['wgir'] = [[] for i in _n]									#
		tVec = []																#
		NPV = []																#
		mNPV = []																#
		s0 = np.array([0]*8+[1]*4)												# (TODO: make these 3 flexible)
		t0 = 0																	#
		eps = .04																#

		for i in range(iterations):													# For each iteration....
			self.curData = EclSum(s.format(i,i+1))									# Load Summary 
			info = self.calculateNPV(t0=t0,mnpv=False)								# Calculate NPV
			infoMod = self.calculateNPV(t0=t0,s0=s0,eps=eps,mnpv=True)				# Calculate MNPV
			t = self.curData.get_days()												# Get time array...
			tVec.extend(list(t))													# ... and append it to existing data
			NPV.extend(info['NPV'])													# Do the same with NPV...
			mNPV.extend(infoMod['NPV'])												# ... and MNPV as well.

			if i == 0:																#
				wellNPV = info['wellNPVs']											# Construct well NPV,
				wellMNPV = infoMod['wellNPVs']										# well MNPV,
				wellSat = info['s']													# and well Oil Saturation arrays
			else:																	#
				wellNPV = np.concatenate((wellNPV,info['wellNPVs']),axis=1)			#
				wellMNPV = np.concatenate((wellMNPV,infoMod['wellNPVs']),axis=1)	#
				wellSat = np.concatenate((wellSat,info['s']),axis=1)				#

			t0 = t[-1]
			for well in self.curData.wells():										#
				_j = self.wellIndex(well)											#
				s0[_j] = self.getSaturation(well)[-1]								#
				wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)				# Build rates info.
				self.rates['wopr'][_j].extend(list(wopr))							#
				self.rates['woir'][_j].extend(list(woir))							#
				self.rates['wwpr'][_j].extend(list(wwpr))							#
				self.rates['wwir'][_j].extend(list(wwir))							#
				self.rates['wgpr'][_j].extend(list(wgpr))							#	
				self.rates['wgir'][_j].extend(list(wgir))							#
		'''
			From now on, plot rates and NPVs info.
		'''
		plt.clf()
		plt.plot(np.array(tVec),np.array(NPV))
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('NPV (USD)')
		plt.title('NPV')
		plt.savefig('FIG/NPV.png',dpi=200)

		plt.clf()
		plt.plot(np.array(tVec),np.cumsum(NPV))
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('NPV (USD)')
		plt.title('Cumulative NPV')
		plt.savefig('FIG/NPVCumulative.png',dpi=200)	

		plt.clf()
		plt.plot(np.array(tVec),np.array(mNPV))
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('MNPV (USD)')
		plt.title('Modified NPV')
		plt.savefig('FIG/MNPV.png',dpi=200)

		plt.clf()
		plt.plot(np.array(tVec),np.cumsum(mNPV))
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('MNPV (USD)')
		plt.title('Cumulative Modified NPV')
		plt.savefig('FIG/MNPVCumulative.png',dpi=200)

		for rate in ['wopr','wwpr']:
			plt.clf()
			for well in self.wells['prod']:
				plt.plot(np.array(tVec),np.array(self.rates[rate][self.wellIndex(well)]),label=well)
			plt.title(rate.upper())
			plt.grid(True)
			plt.legend(loc='best')
			plt.savefig('FIG/{}.png'.format(rate.upper()), dpi=200)

		plt.clf()
		for well in self.wells['prod']:
			plt.plot(np.array(tVec),wellNPV[self.wellIndex(well),:],label=well)
		plt.title('Well NPVs')
		plt.legend(loc='best')
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('NPV (USD)')
		plt.savefig('FIG/WellNPV.png',dpi=200)

		plt.clf()
		for well in self.wells['prod']:
			plt.plot(np.array(tVec),wellMNPV[self.wellIndex(well),:],label=well)
		plt.title('Well Modified NPVs')
		plt.legend(loc='best')
		plt.grid(True)
		plt.xlabel('Time (days)')
		plt.ylabel('NPV (USD)')
		plt.savefig('FIG/WellMNPV.png',dpi=200)

		'''
			.mat file handling.
		'''
		dataMat = {
			'wellNPV' : wellNPV,
			'wellMNPV' : wellMNPV,
			'wellSat' : wellSat,
			't' : np.array(tVec),
			'NPV' : np.array(NPV),
			'MNPV' : np.array(mNPV),
			'wopr' : self.rates['wopr'],
			'wwpr' : self.rates['wwpr'],
			'wwir' : self.rates['wwir']
		}

		sio.savemat('dumpData.mat',dataMat) # Save .mat file

	def directSimulation(self,controls,outputDir='DIRECT'):
		"""Executes a Direct Simulation (no additional algorithm) and saves info to a .mat file.

		Args:
			controls (dict): Schedule controls.
			outputDir: Folder to save data (default: 'DIRECT')
		"""
		fIn = 'BASE/{}UNI.DATA'.format(self.caseName)			#
		fOut = 'UNI{}.DATA'.format(self.caseName)				# File settings
		_x = np.shape(controls['prod'])[0]						#
		StormEclSupport.schedule('UNISCHEDULE.INC','incremental',self.wells,controls,90,_x,('BHP','BHP'))	# Generate Files
		StormEclSupport.configureEclFiles(fIn,fOut,schName='UNISCHEDULE.INC')								#
		tmp = self.dumpName				
		self.dumpName = outputDir 																			# Change Flow's dump directory...
		self.execFlow(fOut)																					# ... execute Flow...
		self.dumpName = tmp 																				# ... and retrieve the original info
		dump = '{}/UNI{}.UNSMRY'.format(outputDir,self.caseName)											#
		self.curData = EclSum(dump)																			# Load resultant summary file
		t = self.curData.get_days()																			# Time array
		_n = len(self.wells['prod'] + self.wells['inje'])													# _n: number of wells

		self.rates['wopr'] = [[] for i in range(_n)]														# Rate initializations
		self.rates['woir'] = [[] for i in range(_n)]
		self.rates['wwpr'] = [[] for i in range(_n)]
		self.rates['wwir'] = [[] for i in range(_n)]
		self.rates['wgpr'] = [[] for i in range(_n)]
		self.rates['wgir'] = [[] for i in range(_n)]	

		for well in self.curData.wells():
			_j = self.wellIndex(well)
			wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)											# Get rates for each well
			self.rates['wopr'][_j].extend(list(wopr))
			self.rates['woir'][_j].extend(list(woir))
			self.rates['wwpr'][_j].extend(list(wwpr))
			self.rates['wwir'][_j].extend(list(wwir))
			self.rates['wgpr'][_j].extend(list(wgpr))
			self.rates['wgir'][_j].extend(list(wgir))	

		s0 = self.s0																			# (TODO: make initial saturation flexible)
		t0 = self.t0
		eps = self.eps
		info, infoMod = self.calculateNPV(t0=t0,modified=False), self.calculateNPV(t0=t0,s0=s0,eps=eps,mnpv=True)	# Calculate NPV and MNPV

		dataMat = {																							# Bundle data into a dictionary
			'wellNPV' : info['wellNPVs'],
			'wellMNPV' : infoMod['wellNPVs'],
			'wellSat' : info['s'],
			't' : np.array(t),
			'NPV' : info['NPV'],
			'MNPV' : infoMod['NPV'],
			'wopr' : self.rates['wopr'],
			'wwpr' : self.rates['wwpr'],
			'wwir' : self.rates['wwir']
		}

		sio.savemat('dumpUniData.mat',dataMat)																# Save data into a .mat file

	def rateDirectSimulation(self,dataName,outputDir='RATE'):
		"""Executes a Direct Simulation (no additional algorithm) from data from a .mat file, uses Rate Control and saves info to a .mat file.
		Usage not recommended as it is.

		Args:
			dataName (dict): Schedule controls .mat file name.
			outputDir: Folder to save data (default: 'RATE')
		"""
		data = sio.loadmat(dataName)
		cP = np.transpose(data['wbhp'])[:,8:12]
		_x = np.shape(cP)[0]
		cI = [8*[60] for i in range(_x)]
		control = {
			'prod' : cP,
			'inje' : cI
		}
		fIn = 'BASE/{}UNI.DATA'.format(self.caseName)
		fOut = 'RATE{}.DATA'.format(self.caseName)
		StormEclSupport.configureEclFiles(fIn,fOut,schName='SCHRATE.INC')
		x = data['t'][0]
		t = [x[i] if i == 0 else x[i]-x[i-1] for i in range(len(x))]
		StormEclSupport.vectorSchedule('SCHRATE.INC',self.wells,control,t,'ORAT')

		tmp = self.dumpName
		self.dumpName = outputDir
		self.execFlow(fOut)
		self.dumpName = tmp
		dump = '{}/RATE{}.UNSMRY'.format(outputDir,self.caseName)
		self.curData = EclSum(dump)
		t = self.curData.get_days()
		_n = len(self.wells['prod'] + self.wells['inje'])

		self.rates['wopr'] = [[] for i in range(_n)]
		self.rates['woir'] = [[] for i in range(_n)]
		self.rates['wwpr'] = [[] for i in range(_n)]
		self.rates['wwir'] = [[] for i in range(_n)]
		self.rates['wgpr'] = [[] for i in range(_n)]
		self.rates['wgir'] = [[] for i in range(_n)]	

		for well in self.curData.wells():
			_j = self.wellIndex(well)
			wopr,woir,wwpr,wwir,wgpr,wgir = self.genVectors(well)
			self.rates['wopr'][_j].extend(list(wopr))
			self.rates['woir'][_j].extend(list(woir))
			self.rates['wwpr'][_j].extend(list(wwpr))
			self.rates['wwir'][_j].extend(list(wwir))
			self.rates['wgpr'][_j].extend(list(wgpr))
			self.rates['wgir'][_j].extend(list(wgir))	

		s0 = np.array([0]*8+[1]*4)
		t0 = 0
		eps = .04
		info, infoMod = self.calculateNPV(t0=t0,modified=False), self.calculateNPV(t0=t0,s0=s0,eps=eps,mnpv=True)

		dataMat = {
			'wellNPV' : info['wellNPVs'],
			'wellMNPV' : infoMod['wellNPVs'],
			'wellSat' : info['s'],
			't' : np.array(t),
			'NPV' : info['NPV'],
			'MNPV' : infoMod['NPV'],
			'wopr' : self.rates['wopr'],
			'wwpr' : self.rates['wwpr'],
			'wwir' : self.rates['wwir']
		}

		sio.savemat('dumpRateData.mat',dataMat)

class StormBackendException(Exception):
	"""Class for specific Storm Backend Exceptions.
	"""
	pass


def caseConf():
	"""Sets an example case.

	Args:
		None:

	Returns:
		k: Storm Backend instance.
		C: Controls
	"""
	def stb():						# STB constant
		return 0.158987294928

	costs = {						# Costs initialization
		'OilProd' : 40./stb(),
		'WaterProd' : 6./stb(),
		'GasProd' : 0.0,
		'OilInj' : 0.0,
		'WaterInj' : 2./stb(),
		'GasInj' : 0.0
	} 

		

	pA,pI = [150 for i in range(len(wp))],[235 for i in range(len(wi))]					# Well controls


	caseConfigurations = {																# Configurations for current case
		'costs' : costs,
		'producers' : wp,
		'injectors' : wi,
		'prodControl' : pA,
		'injeControl' : pI,
		'span' : 9,
		'prodConstraints' : (150,300),
		'injeConstraints' : (0,235)
	}

	k = StormBackend('EGG',**caseConfigurations) # Case initialization
	cI = [pI for i in range(19)]
	#cP = [[381.535, 380.38, 380.95750000000004, 380.38], [379.24579, 382.66228, 378.671755, 378.09772], [381.52126474, 380.36630632000004, 380.94378553, 380.36630632], [379.23213715156, 378.08410848208, 378.65812281682, 378.65465794156], [381.5075299744693, 381.13872085111825, 380.93007155372095, 380.9265858892094], [383.79657515431614, 376.58168634878945, 383.2156519830433, 383.21214540454463], [386.09935460524207, 374.02854918380064, 385.51494589494155, 385.5114182769719], [381.35647974706694, 376.27272047890347, 387.8280355703112, 387.8244867866337], [377.91426382951749, 378.53035680177686, 389.5732617303776, 390.15143370735353], [374.13492002175207, 380.80153894258751, 390.74198151556874, 392.49234230959763], [372.45131288165419, 380.80153894258751, 390.74198151556874, 370.69975983921228], [370.21660500436428, 380.80153894258751, 367.31586237627135, 368.47556128017698], [372.43790463439046, 383.08634817624306, 369.51975755052899, 365.61658599610814], [370.20327720658412, 385.3848662653005, 364.90218809383447, 363.42288648013147], [369.64797229077425, 387.67902391783502, 364.27088762642273, 362.87775215041131], [367.43008445702958, 385.35294977432801, 366.4565129521813, 360.70048563750885], [365.22550395028742, 383.61886150034354, 364.25777387446823, 358.53628272368377], [363.03415092658571, 381.31714833134146, 362.0722272312214, 356.38506502734168]]
	cP = [pA for i in range(19)]
	C = {																							# Controls (TODO: make flexible)
		'prod' : cP, 'inje' : cI
	}
	return k, C


# Esse eh a funcao central do programa
def mnpoConf():
	"""Sets an example case and runs a MNPO with it.

	Args:
		None.

	Returns:
		k: Storm Backend instance.
		ct: MNPO info
	"""
	def stb():						# STB Constant
		return 0.158987294928

	costs = {						# Costs initialization
		'OilProd' : 45./stb(),
		'WaterProd' : 6./stb(),
		'GasProd' : 0.0,
		'OilInj' : 0.0,
		'WaterInj' : 2/stb(),
		'GasInj' : 0.0
	} 

	#TODO: Egg Model tem 4 produtores e 8 injetores
	numInj = 8
	numProd = 4
	wp,wi = ['PROD-%d'%i for i in range(1,numProd)],['INJ-%d'%i for i in range(1,numInj)]	# Well Names
	pA,pI = [150 for i in range(len(wp))],[235 for i in range(len(wi))]					# Well Controls

	StormEclSupport.eraseAll()																				# Erase support files


	caseConfigurations = {																# Configurations for current case
		'costs' : costs,
		'producers' : wp,
		'injectors' : wi,
		'prodControl' : pA,
		'injeControl' : pI,
		'span' : 7,							# OLYMPUS: 7 -> TODO: ver oq eh esse span (tem haver com possibilidades)
		'prodConstraints' : (150,300),
		'injeConstraints' : (0,235),
		'maxIterations' : 80,				# OLYMPUS: 80
		'timeStep' : 90,
		't0' : 0,
		'eps' : .04
	}


	k = StormBackend('EGG',**caseConfigurations)	# Case init
	k.MNPOAlgorithm((True,'true'))																			# Run MNPO
	ct = k.collectInfo()																					# Get info
	sio.savemat('mnpodata.mat',ct)

	tup = np.shape(ct['bhp'])
	tupi = np.shape(ct['injRates'])
	
	C = {
		'prod' : [[ct['bhp'][i,j] for j in range(tup[1])] for i in range(tup[0])],
		'inje' : [[ct['injRates'][i,j] for j in range(tupi[1])] for i in range(tupi[0])]
	}
	#k.directSimulation(C)																					# Direct simulation with MNPO control (TODO: fix it)
	return k, ct

def directSimulation():
	"""Direct Simulation Function
	Args:
		None.
	"""

	def stb():						# STB Constant
		return 0.158987294928

	costs = {						# Costs initialization
		'OilProd' : 45./stb(),
		'WaterProd' : 6./stb(),
		'GasProd' : 0.0,
		'OilInj' : 0.0,
		'WaterInj' : 2/stb(),
		'GasInj' : 0.0
	} 

	#TODO: Egg Model tem 4 produtores e 8 injetores
	numInj = 8
	numProd = 4
	wp,wi = ['PROD-%d'%i for i in range(1,numProd)],['INJ-%d'%i for i in range(1,numInj)]	# Well Names
	pA,pI = [150 for i in range(len(wp))],[235 for i in range(len(wi))]					# Well Controls

	caseConfigurations = {																# Configurations for current case
		'costs' : costs,
		'producers' : wp,
		'injectors' : wi,
		'prodControl' : pA,
		'injeControl' : pI,
		'span' : 7,							# OLYMPUS: 7
		'prodConstraints' : (150,300),
		'injeConstraints' : (0,235),
		'maxIterations' : 80,				# OLYMPUS: 80
		'timeStep' : 90,
		't0' : 0,
		'eps' : .04
	}

	StormEclSupport.eraseAll()																				# Erase support files
	k = StormBackend('EGG',**caseConfigurations)	# Case init
	control = {
		'prod': np.array([pA for i in range(80)]),
		'inje': np.array([pI for i in range(80)])
	}
	k.directSimulation(control)


def rateEx():
	"""Rate simulation example (do not use for now).
	"""
	k, C = caseConf()
	k.rateDirectSimulation('mnpodata.mat')

############################################################################
### INJECTION CONTROL

def identify(q, q_hat, theta, phi, K, P, forg_factor):
	"""Injection identification via recursive method.

	"""
	Pphi = np.dot(P, phi)
	den = forg_factor + np.dot(phi, Pphi)

	K = Pphi / den
	theta = theta + np.dot(K, q - q_hat)	
	P = (P - np.dot(Pphi, Pphi) ) / (den * forg_factor)

	return theta, K, P

def predict(theta, q, u, dt):
	"""Predicts Rate Model.

	"""
	phi = np.array([q, u*dt])
	q_hat = np.dot(phi, theta)

	return q_hat

def calc_inj(data, data_old, theta_w, P_w, K_w, q_hat_w, theta_l, P_l, K_l, q_hat_l, fator, q_max_w, q_max_l, it, dt = 90.0):
	"""

	"""
	
	# Field data rates extraction
	fwir0, fwir = data_old.numpy_vector('FWIR'), data.numpy_vector('FWIR')
	fwpr0, fopr0 = data_old.numpy_vector('FWPR'), data_old.numpy_vector('FOPR')
	fwpr, fopr = data.numpy_vector('FWPR'), data.numpy_vector('FOPR')

	
	# Last and actual injections used
	inj_old, inj = fwir0[-1], fwir[-1]

	# Actual and previous productions
	q_w, q_l = fwpr[-1], fwpr[-1] + fopr[-1]
	q_w_old, q_l_old = fwpr0[-1], fwpr0[-1] + fopr0[-1]


	# Regressors
	phi_w = np.array([q_w_old, inj_old*dt])
	phi_l = np.array([q_l_old, inj_old*dt])
	
	# Identify water production model
	if it == 2:
		q_hat_w = q_w_old
		q_hat_l = q_l_old


	theta_w, K_w, P_w = identify(q_w, q_hat_w, theta_w, phi_w, K_w, P_w, fator)

	# Identify fluid production model
	theta_l, K_l, P_l = identify(q_l, q_hat_l, theta_l, phi_l, K_l, P_l, fator)

	# Predict oil and fluid production
	q_hat_w = predict(theta_w, q_w, inj, dt)
	q_hat_l = predict(theta_l, q_l, inj, dt)

	
	# Check if predicted poduction is higher than max
	# if yes, lower injection
	inj = inj / max(q_hat_w/q_max_w, q_hat_l/q_max_l, 1)	

	return inj, theta_w, K_w, P_w, q_hat_w, theta_l, K_l, P_l, q_hat_l

###############################################################################

if __name__ == '__main__':
	"""Main (just execute a MNPO and saves data)
	"""
	k, ct = mnpoConf()
	sio.savemat('MATFILES/somedata.mat',ct)
