from global_variables import *
import numpy as np
import math

class TripodGait:

    def __init__(self):
        
        self.gaitLegNo = np.zeros(6)
        self.gaitLegNo[RIGHT_FRONT] = 0
        self.gaitLegNo[LEFT_MIDDLE] = 0
        self.gaitLegNo[RIGHT_REAR]  = 0
        self.gaitLegNo[LEFT_FRONT]  = 2
        self.gaitLegNo[RIGHT_MIDDLE]= 2
        self.gaitLegNo[LEFT_REAR]   = 2
        self.initialPos = np.zeros((6, 3))
        self.nextPos = np.zeros((6, 3))

        self.pushSteps = 2
        self.stepsInCycle = 4
        self.tranTime = 65
        self.step = 0
        self.liftHeight = 25

        self.speed = [0, 0, 0]
        self.gaits = np.zeros((6, 4))
        self.cycleTime = (self.stepsInCycle * self.tranTime) / 1000.0


    def generateGait(self, leg, speed):
        #self.gaits = np.zeros((6, 4))
         # Leg Up, middle position
        print(f'step={self.step}, gaitLegNo={self.gaitLegNo[leg]}')
        if(self.step == self.gaitLegNo[leg]):
            self.gaits[leg][0] = 0
            self.gaits[leg][1] = 0
            self.gaits[leg][2] = -self.liftHeight
            self.gaits[leg][3] = 0            
        # Leg Down position
        elif (((self.step == self.gaitLegNo[leg] + 1) or (self.step == self.gaitLegNo[leg] - (self.stepsInCycle-1))) and (self.gaits[leg][2] < 0)):
            self.gaits[leg][0] = (speed[0] * self.cycleTime * self.pushSteps) / (2 * self.stepsInCycle)
            self.gaits[leg][1] = (speed[1] * self.cycleTime * self.pushSteps) / (2 * self.stepsInCycle)
            self.gaits[leg][2] = 0
            self.gaits[leg][3] = (speed[2] * self.cycleTime * self.pushSteps) / (2 * self.stepsInCycle)
        # Move body forward
        else:
            self.gaits[leg][0] = self.gaits[leg][0] - ((speed[0] * self.cycleTime) / self.stepsInCycle)
            self.gaits[leg][1] = self.gaits[leg][1] - ((speed[1] * self.cycleTime) / self.stepsInCycle)
            self.gaits[leg][2] = 0
            self.gaits[leg][3] = self.gaits[leg][3] - ((speed[2] * self.cycleTime) / self.stepsInCycle)
        return self.gaits[leg]
    
    # Body IK solver: compute where the legs should be
    def bodyIK(self, X, Y, Z, Xdisp, Ydisp, Zrot):
        pos = [0, 0, 0]
        cosB = math.cos(bodyRotX)
        sinB = math.sin(bodyRotX)
        cosG = math.cos(bodyRotY)
        sinG = math.sin(bodyRotY)
        cosA = math.cos(bodyRotZ + Zrot)
        sinA = math.sin(bodyRotZ + Zrot)

        totalX = X + Xdisp + bodyPosX
        totalY = Y + Ydisp + bodyPosY

        print('TotalX = ', totalX)
        print('TotalY = ', totalY)

        pos[0] = totalX - (totalX*cosG*cosA + totalY*sinB*sinG*cosA + Z*cosB*sinG*cosA - totalY*cosB*sinA + Z*sinB*sinA) + bodyPosX
        pos[1] = totalY - (totalX*cosG*sinA + totalY*sinB*sinG*sinA + Z*cosB*sinG*sinA + totalY*cosB*cosA - Z*sinB*cosA) + bodyPosY
        pos[2] = Z - (-totalX*sinG + totalY*sinB*cosG + Z*cosB*cosG)
        return pos

    def doIK(self):       
        legpos = [0,0,0]
        print(f"================[Step: {self.step}]======================")
        gait = self.generateGait(RIGHT_FRONT, self.speed)
        bodypos = self.bodyIK(self.initialPos[RIGHT_FRONT][0] + gait[0], self.initialPos[RIGHT_FRONT][1] + gait[1], self.initialPos[RIGHT_FRONT][2] + gait[2], X_COXA, Y_COXA, gait[3])
        legpos[0] = self.initialPos[RIGHT_FRONT][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[RIGHT_FRONT][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[RIGHT_FRONT][2] + gait[2] + bodypos[2]
        self.nextPos[RIGHT_FRONT] = legpos
        print("RF : ", gait)

        gait = self.generateGait(LEFT_MIDDLE, self.speed)
        bodypos = self.bodyIK(self.initialPos[LEFT_MIDDLE][0] + gait[0], self.initialPos[LEFT_MIDDLE][1] + gait[1], self.initialPos[LEFT_MIDDLE][2] + gait[2], 0, -Y_COXA, gait[3])
        legpos[0] = self.initialPos[LEFT_MIDDLE][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[LEFT_MIDDLE][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[LEFT_MIDDLE][2] + gait[2] + bodypos[2]
        self.nextPos[LEFT_MIDDLE] = legpos
        print("LM : ", gait)

        gait = self.generateGait(RIGHT_REAR, self.speed)
        bodypos = self.bodyIK(self.initialPos[RIGHT_REAR][0] + gait[0], self.initialPos[RIGHT_REAR][1] + gait[1], self.initialPos[RIGHT_REAR][2] + gait[2], -X_COXA, Y_COXA, gait[3])
        legpos[0] = self.initialPos[RIGHT_REAR][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[RIGHT_REAR][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[RIGHT_REAR][2] + gait[2] + bodypos[2]
        self.nextPos[RIGHT_REAR] = legpos
        print("RR : ", gait)

        gait = self.generateGait(LEFT_FRONT, self.speed)
        bodypos = self.bodyIK(self.initialPos[LEFT_FRONT][0] + gait[0], self.initialPos[LEFT_FRONT][1] + gait[1], self.initialPos[LEFT_FRONT][2] + gait[2], X_COXA, -Y_COXA, gait[3])
        legpos[0] = self.initialPos[LEFT_FRONT][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[LEFT_FRONT][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[LEFT_FRONT][2] + gait[2] + bodypos[2]
        self.nextPos[LEFT_FRONT] = legpos
        print("LF : ", gait)

        gait = self.generateGait(RIGHT_MIDDLE, self.speed)
        bodypos = self.bodyIK(self.initialPos[RIGHT_MIDDLE][0] + gait[0], self.initialPos[RIGHT_MIDDLE][1] + gait[1], self.initialPos[RIGHT_MIDDLE][2] + gait[2], 0, Y_COXA, gait[3])
        legpos[0] = self.initialPos[RIGHT_MIDDLE][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[RIGHT_MIDDLE][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[RIGHT_MIDDLE][2] + gait[2] + bodypos[2]
        self.nextPos[RIGHT_MIDDLE] = legpos
        print("RM : ", gait)

        gait = self.generateGait(LEFT_REAR, self.speed)
        bodypos = self.bodyIK(self.initialPos[LEFT_REAR][0] + gait[0], self.initialPos[LEFT_REAR][1] + gait[1], self.initialPos[LEFT_REAR][2] + gait[2], -X_COXA, -Y_COXA, gait[3])
        legpos[0] = self.initialPos[LEFT_REAR][0] + gait[0] + bodypos[0]
        legpos[1] = self.initialPos[LEFT_REAR][1] + gait[1] + bodypos[1]
        legpos[2] = self.initialPos[LEFT_REAR][2] + gait[2] + bodypos[2]
        self.nextPos[LEFT_REAR] = legpos
        print("LR : ", gait)
        
        self.step = (self.step + 1) % self.stepsInCycle



if __name__ == '__main__':
    gait = TripodGait()
    bodyRotX = 0
    bodyRotY = 0
    bodyRotZ = 0
    bodyPosX = 0
    bodyPosY = 0
    req = gait.bodyIK(0, 60, 80, X_COXA, Y_COXA, math.radians(10))
    print(req)