import time
import os
import keyboard
import math
from coppeliasim_zmqremoteapi_client import *

client = RemoteAPIClient()
sim = client.require('sim')
scene_path = os.getcwd() + '/scene/KRPAI_mobile_robot_Simulation.ttt'

# Function to read ultrasonic sensors
def read_ultrasonic_sensors():
    distances = []
    for _sensor in sensor_handles:
        distance = sim.readProximitySensor(_sensor)[1] * 100
        if distance == 0:
            distance = 400            
        distances.append(distance)
    return distances
    
def robot_init():
    global motor_handles, sensor_handles
    sensor_handles = [sim.getObjectHandle('/HEXA4R/ultrasonic_sensor' + str(i + 1)) for i in range(12)]
    motor_handles = [sim.getObjectHandle('/HEXA4R/motor_' + str(i + 1)) for i in range(4)]

def robot_move(left_speed, right_speed):
    left_speed = left_speed * math.pi / 180
    right_speed = right_speed * math.pi / 180
    # set motor speeds
    sim.setJointTargetVelocity(motor_handles[0], left_speed)
    sim.setJointTargetVelocity(motor_handles[1], left_speed)
    sim.setJointTargetVelocity(motor_handles[2], right_speed)
    sim.setJointTargetVelocity(motor_handles[3], right_speed)



if __name__ == '__main__':
    # load scene
    # sim.loadScene(scene_path)
    sim.startSimulation()
    robot_init()

    room = 'room2'
    save_filename = f'{room}.csv'

    with open(save_filename, 'w') as f:

        while True:       
            dist = read_ultrasonic_sensors()
            print(f'left_dist={round(dist[9], 2)}, rear_dist={round(dist[6])}')

            if keyboard.is_pressed('up'):
                robot_move(90, 90)
                
            elif keyboard.is_pressed('down'):
                robot_move(-90, -90)

            elif keyboard.is_pressed('left'):
                robot_move(-90, 90)

            elif keyboard.is_pressed('right'):
                robot_move(90, -90)
            
            # press 'r' to record the data
            elif keyboard.is_pressed('r'):
                dist_recorded = f'{round(dist[9])},{round(dist[6])},{room}'
                f.write(dist_recorded + '\n')
                print(f'recorded: {dist_recorded}')
            
            elif keyboard.is_pressed('q'):
                break

            else:
                robot_move(0, 0)
                
            time.sleep(0.1)

    sim.stopSimulation()