from coppeliasim_zmqremoteapi_client import *
from global_variables import *
import math
import threading
import time
import math
import numpy as np
from gait import TripodGait

class CoppeliaHexapod:

    def __init__(self, client, RobotName):
        self.sim = client.require('sim')
        self.simIK = client.require('simIK')
        self.stop_sensing_event = threading.Event()
        self.stop_actuation_event = threading.Event()
        
        self.robotHandle = self.sim.getObject('/' + RobotName)
        self.robotBase = self.sim.getObject(f'/{RobotName}/base')
        self.legBase = self.sim.getObject(f'/{RobotName}/legBase')
        self.sizeFactor = self.sim.getObjectSizeFactor(self.robotBase)
        #self.simLegTips = [self.sim.getObject(f'/{RobotName}/footTip' + str(i + 1)) for i in range(6)]
        #self.simLegTargets = [self.sim.getObject(f'/{RobotName}/footTarget' + str(i + 1)) for i in range(6)]

        self.simLegTips = np.zeros(6)
        self.simLegTargets = np.zeros(6)

        # Get handle of tip and target dummy
        self.simLegTips[RIGHT_FRONT] = self.sim.getObject(f'/{RobotName}/footTip1')
        self.simLegTips[RIGHT_MIDDLE] = self.sim.getObject(f'/{RobotName}/footTip2')
        self.simLegTips[RIGHT_REAR] = self.sim.getObject(f'/{RobotName}/footTip3')
        self.simLegTips[LEFT_FRONT] = self.sim.getObject(f'/{RobotName}/footTip6')
        self.simLegTips[LEFT_MIDDLE] = self.sim.getObject(f'/{RobotName}/footTip5')
        self.simLegTips[LEFT_REAR] = self.sim.getObject(f'/{RobotName}/footTip4')

        self.simLegTargets[RIGHT_FRONT] = self.sim.getObject(f'/{RobotName}/footTarget1')
        self.simLegTargets[RIGHT_MIDDLE] = self.sim.getObject(f'/{RobotName}/footTarget2')
        self.simLegTargets[RIGHT_REAR] = self.sim.getObject(f'/{RobotName}/footTarget3')
        self.simLegTargets[LEFT_FRONT] = self.sim.getObject(f'/{RobotName}/footTarget6')
        self.simLegTargets[LEFT_MIDDLE] = self.sim.getObject(f'/{RobotName}/footTarget5')
        self.simLegTargets[LEFT_REAR] = self.sim.getObject(f'/{RobotName}/footTarget4')

        # Create an IK environment:
        self.ikEnv = self.simIK.createEnvironment()
        self.ikGroup = self.simIK.createGroup(self.ikEnv)
        for i in range(6):
            self.simIK.addElementFromScene(self.ikEnv, self.ikGroup, self.robotHandle, self.simLegTips[i], self.simLegTargets[i], self.simIK.constraint_position) 

        self.legPos = np.zeros((6, 3))

        # Coppeliascript 
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.robotHandle)
        self.speed = 0
        self.stepSize = 0.08
        self.stepHeight = 0.02
        self.movementDirection = 0
        self.rotationMode = 0
        self.strength = 1

        # Initialize gait engine:
        self.gait = TripodGait()

        # Summaries the information:
        print("==================================================")
        print(" HEXAPOD ROBOT INITIALIZATION")
        print("==================================================")
        print("Robot Name   : ", RobotName)
        print("simLegTips   : ", self.simLegTips)
        print("simLegTargets: ", self.simLegTargets)
        print("sizefactor   : ", self.sizeFactor)
        print("ScriptHandle : ", self.scriptHandle)
        print("==================================================")


    # Get Current Position of the Legs
    def GetCurrentLegPosition(self):
        # Retrieve all values of the leg position, then convert to mm  
        print('Get current leg positions:')

        # ----- RIGHT MIDDLE ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[RIGHT_MIDDLE], self.legBase)
        self.legPos[RIGHT_MIDDLE][x] = pos[x] * 1000
        self.legPos[RIGHT_MIDDLE][y] = pos[y] * 1000 - M_COXA
        self.legPos[RIGHT_MIDDLE][z] = pos[z] * 1000
        print('RM :', self.legPos[RIGHT_MIDDLE])

        # ----- LEFT MIDDLE ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[LEFT_MIDDLE], self.legBase)
        self.legPos[LEFT_MIDDLE][x] = pos[x] * 1000
        self.legPos[LEFT_MIDDLE][y] = pos[y] * 1000 + M_COXA
        self.legPos[LEFT_MIDDLE][z] = pos[z] * 1000
        print('LM :', self.legPos[LEFT_MIDDLE])

        # ----- RIGHT FRONT ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[RIGHT_FRONT], self.legBase)
        self.legPos[RIGHT_FRONT][x] = pos[x] * 1000 - X_COXA
        self.legPos[RIGHT_FRONT][y] = pos[y] * 1000 - Y_COXA
        self.legPos[RIGHT_FRONT][z] = pos[z] * 1000
        print('RF :', self.legPos[RIGHT_FRONT])

        # ----- LEFT FRONT ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[LEFT_FRONT], self.legBase)
        self.legPos[LEFT_FRONT][x] = pos[x] * 1000 - X_COXA
        self.legPos[LEFT_FRONT][y] = pos[y] * 1000 + Y_COXA
        self.legPos[LEFT_FRONT][z] = pos[z] * 1000
        print('LF :', self.legPos[LEFT_FRONT])

        # ----- RIGHT REAR ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[RIGHT_REAR], self.legBase)
        self.legPos[RIGHT_REAR][x] = pos[x] * 1000 + X_COXA
        self.legPos[RIGHT_REAR][y] = pos[y] * 1000 - Y_COXA
        self.legPos[RIGHT_REAR][z] = pos[z] * 1000
        print('RR :', self.legPos[RIGHT_REAR])

        # ----- LEFT REAR ----- #
        pos = self.sim.getObjectPosition(self.simLegTips[LEFT_REAR], self.legBase)
        self.legPos[LEFT_REAR][x] = pos[x] * 1000 + X_COXA
        self.legPos[LEFT_REAR][y] = pos[y] * 1000 + Y_COXA
        self.legPos[LEFT_REAR][z] = pos[z] * 1000
        print('LR :', self.legPos[LEFT_REAR])

        return self.legPos

    def SetInitialIK(self, leg, pos):
        pass

    # Set Robot Leg Position
    def SetLegPosition(self, leg, pos):
        if leg == RIGHT_FRONT:
            pos[x] = pos[x] + X_COXA
            pos[y] = pos[y] + Y_COXA
            pos[z] = pos[z]

        elif leg == RIGHT_MIDDLE:
            pos[x] = pos[x]
            pos[y] = pos[y] + M_COXA
            pos[z] = pos[z]
        
        elif leg == RIGHT_REAR:
            pos[x] = pos[x] - X_COXA
            pos[y] = pos[y] + Y_COXA
            pos[z] = pos[z]

        elif leg == LEFT_FRONT:
            pos[x] = pos[x] + X_COXA
            pos[y] = pos[y] - Y_COXA
            pos[z] = pos[z]

        elif leg == LEFT_MIDDLE:
            pos[x] = pos[x]
            pos[y] = pos[y] - M_COXA
            pos[z] = pos[z]

        elif leg == LEFT_REAR:
            pos[x] = pos[x] - X_COXA
            pos[y] = pos[y] - Y_COXA
            pos[z] = pos[z]

        # Convert to mm to m
        pos = [pos[i] / 1000 for i in range(3)]
        self.sim.setObjectPosition(self.simLegTargets[leg], pos, self.legBase)
        self.simIK.handleGroup(self.ikEnv, self.ikGroup,{'syncWorlds': True})

    # Set Initial Position of Robot:
    def SetInitialLegPosition(self, X, Y, Z):
        self.SetLegPosition(RIGHT_FRONT, [X, Y, Z])  
        self.gait.initialPos[RIGHT_FRONT] = [X, Y, Z]      

        self.SetLegPosition(RIGHT_MIDDLE, [0, Y + 10, Z])
        self.gait.initialPos[RIGHT_MIDDLE] = [0, Y + 10, Z]

        self.SetLegPosition(RIGHT_REAR, [-X, Y, Z])
        self.gait.initialPos[RIGHT_REAR] = [-X, Y, Z]

        self.SetLegPosition(LEFT_FRONT,[X, -Y, Z])
        self.gait.initialPos[LEFT_FRONT] = [X, -Y, Z]

        self.SetLegPosition(LEFT_MIDDLE,[0, -Y - 10, Z])
        self.gait.initialPos[LEFT_MIDDLE] = [0, -Y - 10, Z]

        self.SetLegPosition(LEFT_REAR,[-X, -Y, Z])
        self.gait.initialPos[LEFT_REAR] = [-X, -Y, Z]

        self.sim.callScriptFunction('SetInitialPos', self.scriptHandle)

    # ---------- Set Robot Speed -------------- #
    def SetSpeed(self, Xspeed, Yspeed, Rspeed):
        self.gait.speed = [Xspeed, Yspeed, Rspeed * DEG_TO_RAD]

    def interpolate_step(self):
        self.SetLegPosition(RIGHT_FRONT, self.gait.nextPos[RIGHT_FRONT])
        self.SetLegPosition(LEFT_MIDDLE, self.gait.nextPos[LEFT_MIDDLE])
        self.SetLegPosition(RIGHT_REAR, self.gait.nextPos[RIGHT_REAR])
        self.SetLegPosition(LEFT_FRONT, self.gait.nextPos[LEFT_FRONT])
        self.SetLegPosition(RIGHT_MIDDLE, self.gait.nextPos[RIGHT_MIDDLE])
        self.SetLegPosition(LEFT_REAR, self.gait.nextPos[LEFT_REAR])

    def interpolate(self):
        for i in range (100):
            self.gait.doIK()
            self.interpolate_step()
            time.sleep(0.1)
            
            
    # Enable IK in sim scene
    def enableSceneIK(self, state):
        ret = self.sim.callScriptFunction('enableIK', self.scriptHandle, state)
        if not ret == 0:
            print('Failed to execute call_script_function \'enableIK\'')

    def callback(self, m, vel, accel):
        self.sim.setObjectMatrix(self.legBase, self.robotBase, m)
        #self.simIK.handleGroup(self.ikEnv, self.ikGroup,{'syncWorlds': True})

    def MoveToPose(self, pos, euler, vel, accel):
        vel = vel / 1000
        accel = accel / 1000
        pos = [pos[0] / 1000, pos[1] / 1000, pos[2] / 1000]
        euler = [euler[0] * DEG_TO_RAD, euler[1] * DEG_TO_RAD, euler[2] * DEG_TO_RAD]
        startPose = self.sim.getObjectMatrix(self.legBase, self.robotBase)
        goalPose = self.sim.buildMatrix(pos, euler)        
        print(f"vel: {vel} accel: {accel}")
        self.sim.moveToPose(-1, startPose,[vel],[accel],[2],goalPose,self.callback,None,[1,1,1,0.5])

    def ReadProximitySensor(self):
        self.proximityHandle = [self.sim.getObject('/PhantomX/distance' + str(i + 1)) for i in range(10)]
        self.distance = [0 for i in range(10)]        
        for i in range(10):
            recv =  self.sim.readProximitySensor(self.proximityHandle[i])
            self.distance[i] = recv[1] * 1000
        return self.distance
    
    # ----------------------------------------------- #
    def sys_sensing(self):
        
        while not self.stop_sensing_event.is_set():
            #distance = self.ReadProximitySensor()
            #print(distance)
            time.sleep(0.1)

        print('Thread sys_sensing() is stopped!')
        
    def sys_actuation(self):
        while not self.stop_actuation_event.is_set():
            time.sleep(0.1)

        print('Thread sys_actuation() is stopped!')
    # ----------------------------------------------- #


    # Set the robot movement data
    # speed (mm/s), direction (deg)
    def Move(self, speed, direction=0, waitTime=0):
        try:
            self.speed = speed/1000
            self.movementDirection = direction
            ret = self.sim.callScriptFunction('setStepMode', self.scriptHandle, self.speed, self.stepSize, self.stepHeight, self.movementDirection,self.rotationMode, self.strength)
            time.sleep(waitTime)
        except Exception as e:
            print(e)

    # Stop robot movement
    def Stop(self):
        try:
            ret = self.sim.callScriptFunction('setStepMode',self.scriptHandle,0,0,0,0,0,0) 
                                
        except Exception as e:
            print(e)        

    def Rotate(self, speed, rotationMode):
        pass



# ========================= Main Script ==================== #
if __name__ == '__main__':
    client = RemoteAPIClient()
    sim = client.require('sim')
    simIK = client.require('simIK')
    sim.startSimulation()
    print("Start simulation...")

    # Initialize Coppelia Robot:
    robot = CoppeliaHexapod(client, 'hexapod_base1')
    robot.enableSceneIK(False)
    robot.SetInitialLegPosition(55, 80, 80)    
    robot.GetCurrentLegPosition()
    
    robot.SetSpeed(0, 200, -45)
    robot.interpolate()

    



    # --------------- move single leg up -------------- #
    #legPosition = robot.GetCurrentLegPosition()
    #leg = legPosition[RIGHT_MIDDLE]
    #leg[z] = leg[z] - 20
    #robot.SetLegPosition(RIGHT_MIDDLE, leg)

    # ============== main loop ================= #
    #start_time = time.time()
    #while time.time() - start_time < 10:
    #    time.sleep(0.5)

    # Stop process:
    time.sleep(3)
    print("Completed!")
    sim.stopSimulation()   
    print("Stop Simulation!")