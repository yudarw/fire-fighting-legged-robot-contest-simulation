function sysCall_init()
    corout=coroutine.create(coroutineMain)
    antBase=sim.getObject('./base')
    legBase=sim.getObject('./legBase')
    
    simLegTips={}
    simLegTargets={}
    for i=1,6,1 do
        simLegTips[i]=sim.getObject('./footTip'..i)
        simLegTargets[i]=sim.getObject('./footTarget'..i)
    end
    initialPos={}
    for i=1,6,1 do
        initialPos[i]=sim.getObjectPosition(simLegTips[i],legBase)
    end
    legMovementIndex={1,3,5,2,4,6}
    stepProgression=0
    realMovementStrength=0
    
    --IK:
    local simBase=sim.getObject('.')
    ikEnv=simIK.createEnvironment()
    -- Prepare the ik group, using the convenience function 'simIK.addElementFromScene':
    ikGroup=simIK.createGroup(ikEnv)
    for i=1,#simLegTips,1 do
        simIK.addElementFromScene(ikEnv,ikGroup,simBase,simLegTips[i],simLegTargets[i],simIK.constraint_position)
    end
    
    movData={}
    movData.vel=0.5
    movData.amplitude=0.16
    movData.height=0.04
    movData.dir=0
    movData.rot=0
    movData.strength=0
    ikEnabled=true
end

function sysCall_actuation()
    if coroutine.status(corout)~='dead' then
        local ok,errorMsg=coroutine.resume(corout)
        if errorMsg then
            error(debug.traceback(corout,errorMsg),2)
        end
    end
    
    if ikEnabled then
        dt=sim.getSimulationTimeStep()    
        dx=movData.strength-realMovementStrength
        if (math.abs(dx)>dt*0.1) then
            dx=math.abs(dx)*dt*0.5/dx
        end
        realMovementStrength=realMovementStrength+dx
        
        for leg=1,6,1 do
            sp=(stepProgression+(legMovementIndex[leg]-1)/6) % 1
            print(sp)
            offset={0,0,0}
            if (sp<(1/3)) then
                offset[1]=sp*3*movData.amplitude/2
            else
                if (sp<(1/3+1/6)) then
                    s=sp-1/3
                    offset[1]=movData.amplitude/2-movData.amplitude*s*6/2
                    offset[3]=s*6*movData.height
                else
                    if (sp<(2/3)) then
                        s=sp-1/3-1/6
                        offset[1]=-movData.amplitude*s*6/2
                        offset[3]=(1-s*6)*movData.height
                    else
                        s=sp-2/3
                        offset[1]=-movData.amplitude*(1-s*3)/2
                    end
                end
            end
            md=movData.dir+math.abs(movData.rot)*math.atan2(initialPos[leg][1]*movData.rot,-initialPos[leg][2]*movData.rot)
            offset2={offset[1]*math.cos(md)*realMovementStrength,offset[1]*math.sin(md)*realMovementStrength,offset[3]*realMovementStrength}
            p={initialPos[leg][1]+offset2[1],initialPos[leg][2]+offset2[2],initialPos[leg][3]+offset2[3]}
            sim.setObjectPosition(simLegTargets[leg],legBase,p) -- We simply set the desired foot position. IK is handled after that
        end
        simIK.applyIkEnvironmentToScene(ikEnv,ikGroup)
        
        stepProgression=stepProgression+dt*movData.vel
    end
end


function enableIK(status)
    ikEnabled=status
end

function SetInitialPos()
    for i=1,6,1 do
        initialPos[i]=sim.getObjectPosition(simLegTips[i],legBase)
    end
end

setStepMode=function(stepVelocity,stepAmplitude,stepHeight,movementDirection,rotationMode,movementStrength)
    movData={}
    movData.vel=stepVelocity
    movData.amplitude=stepAmplitude
    movData.height=stepHeight
    movData.dir=math.pi*movementDirection/180
    movData.rot=rotationMode
    movData.strength=movementStrength
    
    return 1
end

function callback(m,vel,accel,auxData)
    sim.setObjectMatrix(auxData[1],auxData[2],m)
end

function moveToPose(obj,relObj,pos,euler,vel,accel)
    local auxData={obj,relObj}
    local mStart=sim.getObjectMatrix(obj,relObj)
    local mGoal=sim.buildMatrix(pos,euler)
    sim.moveToPose(-1,mStart,{vel},{accel},{0.1},mGoal,callback,auxData,{1,1,1,0.1})
end

-- ---------------------------------------------------------
function initSetup()
    sizeFactor=sim.getObjectSizeFactor(antBase)
    stepHeight=0.02*sizeFactor
    maxWalkingStepSize=0.06*sizeFactor
    walkingVel=0.9
end

function Stop()
    setStepMode(walkingVel,maxWalkingStepSize*0.5,stepHeight,0,0,0)
end

function MoveForward(time)
    setStepMode(0.9,0.06,0.02,0,0,1)
    sim.wait(time)
end

function MoveForward2(speed, dir, time)
    setStepMode(speed,0.06,0.02,dir,0,1)
    sim.wait(time)
end

function RotateLeft(time)
    setStepMode(0.8,0.08,0.02,0,-1,1)
    sim.wait(time)
end

function RotateRight(time)
    setStepMode(0.8,0.08,0.02,0,1,1)
    sim.wait(time)
end

function test(param1, param2)
    print(param1)
    print(param2)
    return 'OK'
end

function run_demo_program()
    initSetup()
    MoveForward(15)
    RotateLeft(3)
    MoveForward(20)
    RotateLeft(3)
    MoveForward(15)
    RotateRight(3.2)
    MoveForward(10)
    RotateRight(7.2)
    MoveForward(9)
    RotateLeft(3.6)
    MoveForward(14)
    MoveForward2(0.9,-30,4)
    MoveForward(22)
    RotateRight(3.1)
    MoveForward(8)
    RotateLeft(6.5)
    MoveForward(29)
    Stop()
    
end

-------------------------------------------------
--  Main Program
-- ----------------------------------------------
function coroutineMain()
    --run_demo_program()
end