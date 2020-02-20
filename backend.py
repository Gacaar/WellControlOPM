import os
import glob

"""
	Basic code for OPM Support in order to execute MNPO and other simulations for the STORMS research group.
"""
class StormEclSupport:
	"""
		Class containing basic operations for file handling in order to generate inputs for OPM Flow.
		All of its methods should be static.
	"""
	def __init__(self):
		pass
	
	@staticmethod
	def wconprod(producers,array,controlType='BHP'):
		"""Generates a string of schedule control for the producer wells.

		Args:
			producers (list): Well names
			array (list): Well control
			controlType (str): Control Type (default: 'BHP'). Can be:
				'BHP','ORAT'

		Returns:
			str: A string containing the schedule control, in ECLIPSE syntax.
		"""
		s = "WCONPROD\n"
		n = len(array)
		for i in range(n):
			if controlType == 'BHP':
				s += "  '%s' 'OPEN' 'BHP' 5700 10000 1* 88000 1* %.6f/\n" % (producers[i],array[i])
			elif controlType == 'ORAT':
				s += "  '%s' 'OPEN' 'ORAT' %.6f 5*/\n" % (producers[i],array[i])
		s += '/'
		return s

	@staticmethod
	def wconinje(injectors,array,controlType='RATE'):
		"""Generates a string of schedule control for the injector wells.

		Args:
			producers (list): Well names
			array (list): Well control
			controlType (str): Control Type (default: 'RATE'). Can be:
				'BHP','RATE'

		Returns:
			str: A string containing the schedule control, in ECLIPSE syntax.
		"""
		s = "WCONINJE\n"
		n = len(array)
		for i in range(n):
			if controlType == 'RATE':
				s += "  '%s' 'WATER' 'OPEN' 'RATE' %.6f 1* 235/\n" % (injectors[i],array[i])
			elif controlType == 'BHP':
				s += "  '%s' 'WATER' 1* 'BHP' 2* %.6f /\n" % (injectors[i],array[i])
		s += '/'
		return s

	'''
		@schType:
			- 'standard'
			- 'incremental'
			- 'iter'
			- 'onestep'
	'''
	@staticmethod
	def schedule(fname,schType,names,controls,dT,it,controlType=('BHP','RATE')): 
		"""Generates a schedule file.

		Args:


		Returns:

		"""
		if schType not in ['standard','incremental','iter','onestep']:
			raise StormEclSupportException('Schedule style not recognized.')

		f = open(fname,'w')

		if schType not in ['standard','incremental'] and it > 0:
			f.write('TSTEP\n')
			s = '  '
			for i in xrange(it):
				s += '{} '.format(dT)
			s += '/'
			f.write(s+'\n\n')

		if schType != 'incremental':
			f.write(StormEclSupport.wconprod(names['prod'],controls['prod'][0],controlType[0])+'\n\n')
			f.write(StormEclSupport.wconinje(names['inje'],controls['inje'][0],controlType[1])+'\n')
			if schType != 'standard':
				f.write('\nTSTEP\n')
				f.write('  {} /\n\n'.format(dT))

				if schType == 'iter':
					f.write(StormEclSupport.wconprod(names['prod'],controls['prod'][1],controlType[0])+'\n\n')
					f.write(StormEclSupport.wconinje(names['inje'],controls['inje'][1],controlType[1])+'\n')
					f.write('\nTSTEP\n')
					f.write('  {} /\n\n'.format(dT))
			else:
				f.write('\nTSTEP\n')
				s = '  '
				for i in xrange(it):
					s += '{} '.format(dT)
				s += '/\n'
				f.write(s+'\n')

		else:
			for i in xrange(it):
				f.write(StormEclSupport.wconprod(names['prod'],controls['prod'][i],controlType[0])+'\n\n')
				f.write(StormEclSupport.wconinje(names['inje'],controls['inje'][i],controlType[1])+'\n')
				f.write('\nTSTEP\n')
				f.write('  {} /\n\n'.format(dT))

		f.write('END')
		f.close()

	@staticmethod
	def vectorSchedule(fname,names,controls,tVector,controlType=('BHP','RATE')):
		f = open(fname,'w')

		for i in xrange(len(tVector)):
			f.write(StormEclSupport.wconprod(names['prod'],controls['prod'][i],controlType[0])+'\n\n')
			f.write(StormEclSupport.wconinje(names['inje'],controls['inje'][i],controlType[1])+'\n')
			f.write('\nTSTEP\n')
			f.write('  {} /\n\n'.format(tVector[i]))

		f.write('END')
		f.close()

	@staticmethod
	def prepareFile(fname):
		dic = {
			'RUNSPEC': [],
			'GRID': [],
			'PROPS': [],
			'REGIONS': [],
			'SOLUTION': [],
			'SUMMARY': [],
			'SCHEDULE': []
		}
		spec = 'RUNSPEC'
		
		f = open(fname,'r')
		S = f.read().split('\n')
		f.close()
		
		for i in range(len(S)):
			if '--' in S[i]:
				#print(S[i],S[i][:S[i].index('--')])
				S[i] = S[i][:S[i].index('--')]
		
		S = [i.strip() for i in S if len(i) > 0]	
		
		for i in range(len(S)):
			if S[i] in dic.keys(): spec = S[i]
			else: dic[spec].append(S[i])
		
		return dic
	
	@staticmethod
	def restartConfig(dic,**kwargs):
		def _blackList(lst,s):
			for i in lst:
				if i in s: return True
			return False

		schName = kwargs['schName']
		rstName = 'PRIOR' if 'rstName' not in kwargs.keys() else kwargs['rstName']
		index = 1 if 'rstIndex' not in kwargs.keys() else kwargs['rstIndex']

		i = 0
		lst = dic['SOLUTION'][:]
		rst = lst[:]
			
		rst.insert(0,"  '{0}'   {1}\t1*\t1* /".format(rstName,index))
		rst.insert(0,'RESTART')
		
		dic['SOLUTION'] = rst[:]
		
		dic = StormEclSupport.scheduleConfig(dic,schName)
			
		dic['SCHEDULE'].insert(0,'SKIPREST')
		return dic

	@staticmethod
	def scheduleConfig(dic,fname):
		dic['SCHEDULE'].append('INCLUDE')
		dic['SCHEDULE'].append(' ' + fname + ' /')
		#if 'INCLUDE' in dic['SCHEDULE']:
		#	i = dic['SCHEDULE'].index('INCLUDE') + 1
		#	dic['SCHEDULE'][i] = '  ' + fname + ' /'
		return dic

	@staticmethod
	def writeEclFile(dic,fname):
		specOrder = ['RUNSPEC','GRID','PROPS','REGIONS','SOLUTION','SUMMARY','SCHEDULE']
		f = open(fname,'w')
		for i in specOrder:
			f.write('--<+> Start of {} Section\n'.format(i))
			f.write(i+'\n')
			f.write('\n')
			for j in dic[i]:
				f.write(j+'\n')
			f.write('--<-> End of {} Section\n'.format(i))
		f.close()

	@staticmethod
	def configureEclFiles(fnameIn,fnameOut,**kwargs):
		"""Configure necessary ECL Files for simulation.

		Args:
			fnameIn (str): Base file name
			fnameOut (str): Output file name

		Key Args:
			schName: Schedule file name
			restart: Indicates a restart simulation
			rstName: Restart file name
			rstIndex: Index to restart from

		Raises:
			StormEclSupport Exception: missing keys for schedule and/or restart info.

		Returns:
			None.
		"""			
		if 'schName' not in kwargs.keys():
			raise StormEclSupportException('Schedule file name must be provided for case file settings.')
		restart = False if 'restart' not in kwargs.keys() else kwargs['restart']

		if restart and not all(x in kwargs.keys() for x in ['rstName','rstIndex']):
			raise StormEclSupportException('Restart parameters must be provided for restart file settings.')
		dic = StormEclSupport.prepareFile(fnameIn)
		if restart:
			dic = StormEclSupport.restartConfig(dic,**kwargs)
		else:
			dic = StormEclSupport.scheduleConfig(dic,kwargs['schName'])
		StormEclSupport.writeEclFile(dic,fnameOut)

	@staticmethod
	def eraseAll():
		"""Erases all the support and dump files generated by STORMS codes.

		Args:
			None.


		Returns:
			None.
		"""
		toErase = glob.glob('DUMP/*.*')
		toErase += [i for i in glob.glob('*.DATA')]
		toErase += [i for i in glob.glob('*.txt')]
		toErase += [i for i in glob.glob('*.INC') if 'SCH' in i]
		for i in toErase: os.remove(i)

class StormEclSupportException(Exception):
	"""Class for specific Storm File Support Exceptions.
	"""
	pass