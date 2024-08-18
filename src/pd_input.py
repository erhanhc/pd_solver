from os import sys
from builtins import IOError
from pandas import read_csv
from yaml import safe_load
import logging
class pd_input:
    def __init__(self,caller):
        self.logger = logging.getLogger('INPUT')
        self.get_cmd_args()
        self.arg_to_data()
        self.data_rearrange(caller)
        pass

    def get_cmd_args(self):
        self.args = sys.argv[1:]
        if not self.args:
            raise IOError('No files given!')
        
    def arg_to_data(self):
        # body,neigh,config 
        #df's:
        self.body = read_csv(self.args[0],index_col=0)
        self.neigh = read_csv(self.args[1],index_col=0)
        
        #yaml_config
        with open(self.args[2],'r') as file:
            self.config = safe_load(file)
    
        self.appliedb= read_csv(self.args[3],index_col=0)
        self.appliedd= read_csv(self.args[4],index_col=0)
        
    def data_rearrange(self,caller):
        caller.coord = self.body[['coord1','coord2']].values
        caller.volumes = self.body['volume'].values
        caller.dii = self.body[['dii1','dii2']].values
        caller.from_to = self.body[['from','to']].values
        
        caller.neighbors = self.neigh[['curr','other']].values
        caller.a = self.neigh['a'].values
        caller.b = self.neigh['b'].values
        caller.d = self.neigh['d'].values
        if self.neigh.columns.__contains__('sc'):
            caller.sc = self.neigh['sc'].values
        
        caller.applied = {}
        caller.applied['bforce'] = {}
        caller.applied['disp'] = {}
        
        caller.applied['bforce']['index'] = self.appliedb.index.values
        caller.applied['bforce']['values']= self.appliedb.values
        
        caller.applied['disp']['index'] = self.appliedd.index.values
        caller.applied['disp']['dof'] = self.appliedd.dof.values
        caller.applied['disp']['appliedd']= self.appliedd.appliedd.values
        
        caller.config = {k:v for k,v in self.config.items()}