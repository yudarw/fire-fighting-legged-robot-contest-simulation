import time
import os
import keyboard
import math
from coppeliasim_zmqremoteapi_client import *
import joblib
import numpy as np

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


def room_data_collection():
    # load scene
    #sim.loadScene(scene_path)
    sim.startSimulation()
    robot_init()

    room = 'room1A'
    save_filename = f'{room}.csv'

    with open(save_filename, 'w') as f:
        i = 0
        while True:       
            dist = read_ultrasonic_sensors()
            #print(f'left_dist={round(dist[9], 2)}, rear_dist={round(dist[6])}')

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
                dist_recorded = f'{round(dist[9])},{round(dist[8])},{round(dist[7])},{round(dist[6])},{round(dist[5])},{round(dist[4])},{round(dist[3])},{round(dist[2])},{room}'
                f.write(dist_recorded + '\n')
                print(f'recorded {i}: {dist_recorded}')
                i += 1
            
            elif keyboard.is_pressed('q'):
                break

            else:
                robot_move(0, 0)
                
            time.sleep(0.2)
    sim.stopSimulation()


def mlp_room_detection():
    # load the model
    mlp = joblib.load('room_detection_mlp_model.pkl')
    scaler = joblib.load('scaler.pkl')
    room_name = ['room1A', 'room1B', 'room2', 'room3', 'room4']
    sim.startSimulation()
    robot_init()
    while True:
        dist = read_ultrasonic_sensors()
        sensor_data = np.array([dist[9], dist[8], dist[7], dist[6], dist[5], dist[4], dist[3], dist[2]]).reshape(1, -1)
        sensor_data = scaler.transform(sensor_data)
        room = mlp.predict(sensor_data)
        print(f'Predicted Room: {room_name[room[0]]}')
        time.sleep(0.5)

if __name__ == '__main__':
    mlp_room_detection()