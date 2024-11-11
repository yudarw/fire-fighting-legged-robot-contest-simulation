from coppeliasim_zmqremoteapi_client import *

class GPS:
    def __init__(self,client,sensor_name):
        self.sim = client.require('sim')
        self.sensor_name = sensor_name
        self.model = self.sim.getObject(f'/{self.sensor_name}')
        self.ref = self.sim.getObject(f'/{self.sensor_name}/reference')
    
    def read(self):
        return self.sim.getObjectPosition(self.ref)