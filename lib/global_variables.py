import math

# -- Robot Type -- #
HEXAPOD = 0

# -- Robot Dimension --#
X_COXA      = 60
Y_COXA      = 30
M_COXA      = 50

# Legs
L_COXA      = 52
L_FEMUR     = 82
L_TIBIA     = 140

# Servo Offset
OFFSET_COXA  = 45
OFFSET_FEMUR = 57.3
OFFSET_TIBIA = -52.92

# Servo ID
RM_TIBIA    = 18
RF_COXA     = 2
LR_TIBIA    = 11
LF_FEMUR    = 3
RF_TIBIA    = 6
RM_FEMUR    = 16
RM_COXA     = 14
RR_COXA     = 8
LF_TIBIA    = 5
LF_COXA     = 1
LR_FEMUR    = 9
RR_FEMUR    = 10
LM_TIBIA    = 17
RF_FEMUR    = 4
LM_FEMUR    = 15
RR_TIBIA    = 12
LM_COXA     = 13
LR_COXA     = 7

LEG_COUNT   = 6

# -- ripple gait move one leg at a time --
RIPPLE          = 0
RIPPLE_SMOOTH   = 3
AMBLE           = 4
AMBLE_SMOOTH    = 5
# -- tripod gaits are only for hexapod -- 
TRIPOD          = 6

# -- Legs Position --
RIGHT_FRONT     = 0
RIGHT_REAR      = 2
LEFT_FRONT      = 5
LEFT_REAR       = 3
RIGHT_MIDDLE    = 1
LEFT_MIDDLE     = 4

STD_TRANSITION  = 98


RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


mins = {222, 225, 159, 164, 279, 158, 223, 229, 159, 156, 272, 155, 226, 233, 158, 157, 271, 157}
maxs = {790, 792, 855, 862, 857, 747, 788, 794, 859, 857, 860, 747, 789, 789, 858, 860, 859, 743}

#LegNames = ["RIGHT_FRONT","RIGHT_REAR","LEFT_FRONT","LEFT_REAR","RIGHT_MIDDLE","LEFT_MIDDLE"]
LegNames = ["RIGHT_FRONT","RIGHT_MIDDLE","RIGHT_REAR","LEFT_REAR","LEFT_MIDDLE","LEFT_FRONT"]

x,y,z = 0,1,2

# --- Gaits Engine --- 
gaits = []         # gait engine output
gaitLegNo = []     # order the step through legs
pushSteps = 0      # how much of the cycle we are on the ground
stepsInCycle = 0   # how much steps in this cycle
step = 0           # Current step
tranTime = 0
liftHeight = 0
cycleTime = 0.     # cycle time in seconds (adjustment from speed to step-size)

# --- IK Engine --- 
endpoints = []
bodyRotX = 0.      # body roll (rad)
bodyRotY = 0.      # body pitch (rad)
bodyRotZ = 0.      # body rotation (mm)
bodyPosX = 0       # body offset (mm)
bodyPosY = 0       # body offset (mm)
Xspeed = 0.         # forward speed (mm/s)
Yspeed = 0.         # sidewaed speed (mm/s)
Rspeed = 0.         # rotation speed (rad/s)

class ik_req_t:
    def __init__(self) -> None:
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.r = 0.
    

class ik_sol_t:
    def __init__(self):
        self.coxa  = 0.
        self.femur = 0.
        self.tibia = 0.    

class ik_speed_t:
    def __init__(self):
        self.x  = 0.
        self.y = 0.
        self.r = 0.



class robot_type:
    def __init__(self):
        self.Hexapod = 0
        self.Quad = 0
    @property
    def Hexapod(self):
        return self.Hexapod


class gait_type():
    # -- ripple gait move one leg at a time --
    RIPPLE          = 0
    RIPPLE_SMOOTH   = 3
    AMBLE           = 4
    AMBLE_SMOOTH    = 5
    # -- tripod gaits are only for hexapod -- 
    TRIPOD          = 6
