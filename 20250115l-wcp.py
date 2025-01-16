from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/My Drive/Colab Notebooks/dimacs2024/evacuate.csv'
df = pd.read_csv(file_path)

import io

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    var_name = row['name']  # Assign the 'name' column value to var_name
    var_value = row['value'] # Assign the 'value' column value to var_value
    # Assign the 'comment' column value to var_comment
    var_comment = row['comment'] 
    globals()[var_name] = var_value  # Dynamically create and assign variables

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from matplotlib import colors as mcolors
import random
import math
import pprint # debug dict to screen
import json # debug dict to file
import zipfile # download all output files at once
# use decision trees to predict the exit a civilian will choose
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Function to compute distance between two points
# In procedural (not OOP) notation
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to draw an exit with a specific open side
def draw_exit(x, y, size, open_side):
    plt.scatter(x, y, color='green', marker='s', s=size)
    if open_side != 0:  # Draw top line if not open
        plt.plot([x - size / 2, x + size / 2], [y + size / 2, y + size / 2], color='orange')
    if open_side != 1:  # Draw right line if not open
        plt.plot([x + size / 2, x + size / 2], [y + size / 2, y - size / 2], color='orange')
    if open_side != 2:  # Draw bottom line if not open
        plt.plot([x + size / 2, x - size / 2], [y - size / 2, y - size / 2], color='orange')
    if open_side != 3:  # Draw left line if not open
        plt.plot([x - size / 2, x - size / 2], [y - size / 2, y + size / 2], color='orange')

# Function to convert from string to boolean
def string_to_bool(s):
    if s.lower() in ('true', '1'):
        return True
    elif s.lower() in ('false', '0'):
        return False
    else:
        raise ValueError("Invalid boolean string")

# Function to update the crowd's positions for each frame
def update_crowd(frame):
    global alarm_values, crowd, decision_tree_results_experiment, decision_tree_results_trial, df_experiment, df_trial, experiment_exit_match, experiment_exit_missed, human_values, labels, scatter_civilians, scatter_firemen, scatter_emeMedTec, text_object, trial_exit_match, trial_exit_missed, trial_values

    # only used after the alarm, but cleaner to assign it here
    elapsed_time = frame - alarm_values['alarm_time']
    # ... (Alarm trigger logic remains the same)
    # Trigger the alarm
    if frame == alarm_values['alarm_time'] and not alarm_values['alarm_triggered']:
        alarm_values['alarm_triggered'] = True

    if alarm_values['alarm_triggered']:
        for individual in crowd:
            individual_velocity = individual['velocity']
            individual_x = individual['position'][0]
            individual_y = individual['position'][1]
            individual_features = [
                individual_x,
                individual_y,
                individual_velocity,
                individual['vision_limit']
            ]
            if (trial_values['run_type'] == 'simulation'):
                exit_predicted = tree_model.predict([individual_features])[0]
                individual['exit_predicted'] = exit_predicted
            if (individual['genre'] == 'civilian'):
                # Civilians move towards their closest exit after the alarm
                # if they can see it
                # Calculate distances to each exit
                dist_to_open1 = distance(individual_x, individual_y, trial_values['open1_x'], trial_values['open1_y'])
                dist_to_open2 = distance(individual_x, individual_y, trial_values['open2_x'], trial_values['open2_y'])
                individual['dist_to_open1'] = dist_to_open1
                individual['dist_to_open2'] = dist_to_open2
                # Can I see the closest exit?
                if (min(dist_to_open1, dist_to_open2) < individual['vision_limit']):
                    # Yes I can see the closest exit, so I move toward it
                    # Choose the closest exit for each individual
                    exit_x = np.where(dist_to_open1 < dist_to_open2, trial_values['open1_x'], trial_values['open2_x'])
                    exit_y = np.where(dist_to_open1 < dist_to_open2, trial_values['open1_y'], trial_values['open2_y'])

                    # Calculate angles towards the chosen exit
                    individual['angle'] = np.arctan2(exit_y - individual_y, exit_x - individual_x)
            else:
                # this individual is a responder
                responder_time = frame - individual['entry_time']
                if (responder_time < responder_deployment_int):
                    # responders initially move toward corner
                    # Calculate attributes of each corner
                    # relative to this responder
                    # SouthWest corner
                    corner_x = 0
                    corner_y = 0
                    corner_SW = {
                        'angle_to_corner': np.arctan2(corner_y - individual_y, corner_x - individual_x),
                        'corner_name': 'SW',
                        'corner_x': 0,
                        'corner_y': 0,
                        'dist_to_corner': distance(corner_x, corner_y, individual_x, individual_y)
                        }
                    # NorthWest corner
                    corner_x = 0
                    corner_y = room_width_inches
                    corner_NW = {
                        'angle_to_corner': np.arctan2(corner_y - individual_y, corner_x - individual_x),
                        'corner_name': 'NW',
                        'corner_x': 0,
                        'corner_y': trial_values['room_width'],
                        'dist_to_corner': distance(corner_x, corner_y, individual_x, individual_y)
                        }
                    # NorthEast corner
                    corner_x = trial_values['room_length']
                    corner_y = trial_values['room_width']
                    corner_NE = {
                        'angle_to_corner': np.arctan2(corner_y - individual_y, corner_x - individual_x),
                        'corner_name': 'NE',
                        'corner_x': 0,
                        'corner_y': 0,
                        'dist_to_corner': distance(corner_x, corner_y, individual_x, individual_y)
                        }
                    # SouthEast corner
                    corner_x = trial_values['room_length']
                    corner_y = 0
                    corner_SE = {
                        'angle_to_corner': np.arctan2(corner_y - individual_y, corner_x - individual_x),
                        'corner_name': 'SE',
                        'corner_x': 0,
                        'corner_y': 0,
                        'dist_to_corner': distance(corner_x, corner_y, individual_x, individual_y)
                        }
                    corner_list = [corner_SW, corner_NW, corner_NE, corner_SE]
                    # select closest corner
                    closest_corner = min(corner_list, key=lambda x:x['dist_to_corner'])
                    # and travel toward it
                    individual['angle'] = closest_corner['angle_to_corner']
                else:
                    # after initial deployment
                    # responder moves toward speed-challenged civilian
                    # these are all of the speed-challenged civilians (only civilians have this speed)
                    slow_people = [person for person in crowd if person['person_type'] == 'challenged']
                    # calculate distance from this responder to each speed-challenged civilian
                    for iSlowCalc in range(len(slow_people)):
                        slow_x = slow_people[iSlowCalc]['position'][0]
                        slow_y = slow_people[iSlowCalc]['position'][1]
                        slow_people[iSlowCalc]['dist_to_responder'] = distance(individual_x, individual_y, slow_x, slow_y)
                        # Calculate distances to each exit for the slow person
                        dist_to_open1_slow = distance(slow_x, slow_y, trial_values['open1_x'], trial_values['open1_y'])
                        dist_to_open2_slow = distance(slow_x, slow_y, trial_values['open2_x'], trial_values['open2_y'])
                        # Choose the closest exit for the slow person
                        exit_x_slow = np.where(dist_to_open1_slow < dist_to_open2_slow, trial_values['open1_x'], trial_values['open2_x'])
                        exit_y_slow = np.where(dist_to_open1_slow < dist_to_open2_slow, trial_values['open1_y'], trial_values['open2_y'])
                        # Assign the closest exit coordinates to the slow person's dictionary
                        slow_people[iSlowCalc]['exit'] = [exit_x_slow, exit_y_slow]
                    # select the closest speed-challenged civilian
                    if slow_people:  # Check if slow_people is not empty
                        closest_slow = min(slow_people, key=lambda x: x['dist_to_responder'])
                        closest_x = closest_slow['position'][0]
                        closest_y = closest_slow['position'][1]
                        exit_x = closest_slow['exit'][0]
                        exit_y = closest_slow['exit'][1]
                        # how far away is he
                        dist_to_slow = distance(closest_x, closest_y, individual_x, individual_y)
                        # if he is already nearby, then lead him to the nearest exit
                        if dist_to_slow < trial_values['diameter_doubled']:
                            # civilian already knows the nearest exit
                            # responder travels toward the nearest exit
                            individual['angle'] = np.arctan2(exit_y - individual_y, exit_x - individual_x)
                        else:
                            # responder travels toward the closest speed-challenged civilian
                            individual['angle'] = np.arctan2(closest_y - individual_y, closest_x - individual_x)
                    else:
                        # If no slow people, direct responders to an exit 
                        individual['angle'] = np.arctan2(trial_values['open1_y'] - individual_y, trial_values['open1_x'] - individual_x) 

            # Increase velocity after the alarm: increase speed
            # but limit to top speed
            # move faster even if I cannot see the closest exit
            individual_faster = individual_velocity * 1.1
            if individual_faster < individual['velocity_limit']:
                individual['velocity'] = individual_faster

    # Update positions based on current velocities and directions
    for individual in crowd:
        # previous position was ( x_1 , y_1 )
        individual_x1 = individual['position'][0]
        individual_y1 = individual['position'][1]
        # candidate new position is ( x_2 , y_2 )
        individual_angle = individual['angle']
        individual_velocity = individual['velocity']
        individual_x2 = individual_x1 + individual_velocity * np.cos(individual_angle)
        individual_y2 = individual_y1 + individual_velocity * np.sin(individual_angle)

        # ... (Wall bouncing and direction change logic, adapted for dictionary access)
        # Bounce off the walls (except at exits)
        # (1 * diameter) doesn't bounce off, people just get stuck
        # see https://3dkingdoms.com/weekly/weekly.php?a=2
        # this might get the angles correct, but doesn't get positions correct
        if individual_x2 - trial_values['diameter_doubled'] < 0:  # Hit left wall
            individual_x2 = trial_values['diameter_doubled']  # Prevent going beyond the wall
            individual_angle = np.pi - individual_angle  # Reflect angle
        elif individual_x2 + trial_values['diameter_doubled'] > trial_values['room_length']:  # Hit right wall
            individual_x2 = trial_values['room_length'] - trial_values['diameter_doubled']  # Prevent going beyond the wall
            individual_angle = np.pi - individual_angle  # Reflect angle

        if individual_y2 - trial_values['diameter_doubled'] < 0:  # Hit bottom wall
            individual_y2 = trial_values['diameter_doubled']  # Prevent going beyond the wall
            individual_angle = -individual_angle  # Reflect angle
        elif individual_y2 + trial_values['diameter_doubled'] > trial_values['room_width']:  # Hit top wall
            # Prevent going beyond the wall
            individual_y2 = trial_values['room_width'] - trial_values['diameter_doubled']  
            individual_angle = -individual_angle  # Reflect angle

        # these should now be correct new positions and angles
        individual['angle'] = individual_angle
        individual['position'][0] = individual_x2
        individual['position'][1] = individual_y2
        
        # Randomly change direction and speed for some individuals (before the alarm)
        if not alarm_values['alarm_triggered']:
            random_float = random.random() 
            change_me = random_float < 0.05
            if change_me:
                individual['angle'] = random_float * 2 * np.pi
                match (individual['genre'], individual['person_type']):
                    case('civilian', 'able'):
#                        individual_values = [item for item in human_values if item['genre'] == 'civilian' and item['person_type'] == 'able]
                        individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'able', human_values))
                    case('civilian', 'challenged'):
                        individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'challenged', human_values))
                    case('responder', 'fireman'):
                        individual_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'fireman', human_values))
                    case('responder', 'medical'):
                        individual_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'medical', human_values))
                    case _:
                        individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'challenged', human_values))
                # Print for evac debugging
                if trial_values['debug']:
                    print(f"Genre: {individual['genre']}, Person Type: {individual['person_type']}")
                    print(f"Individual Values: {individual_values}")
                
                # Check if the key exists and the list is not empty before accessing
                if individual_values and 'velocity_initial' in individual_values[0]: 
                    individual['velocity'] = random_float * individual_values[0]['velocity_initial']
                else:
                    # Handle the case where the key is not found
                    # either set a default velocity or raise an error
                    individual['velocity'] = 10 # setting a default velocity
                    print("Warning: 'velocity_initial' key not found or individual_values list is empty.")

    # ... (Exit and occupancy update logic, adapted for dictionary access)
    # Remove individuals who have reached an exit
    if alarm_values['alarm_triggered']:
        # how many people remain
        people_remaining = len(crowd)
        with open(trial_values['filename_debug'], "a") as f:
            f.write(f"update_crowd({frame}):0100: people_remaining = {people_remaining}\n")
        # is anybody still here
        if people_remaining > 0:
            sorted_people = sorted(crowd, key=lambda x:x['dist_to_open1'])
            with open(trial_values['filename_debug'], "a") as f:
                f.write(f"update_crowd({frame}):0200: len(sorted_people) = {len(sorted_people)}\n")
            iDepart = 0
            removed = False
            while ((iDepart < len(sorted_people)) and (removed == False)):
                with open(trial_values['filename_debug'], "a") as f:
                    f.write(f"update_crowd({frame}):0300:  people_remaining = {people_remaining}\n")
                # who is closest to the open side of exit 1
                if (sorted_people[iDepart]['genre'] == 'civilian'):
                    with open(trial_values['filename_debug'], "a") as f:
                        f.write(f"update_crowd({frame}):0400:(sorted_people[{iDepart}]['genre'] == 'civilian')\n")
                    # are they close enough to depart
                    # (1 * diameter) causes people to get stuck
                    if (sorted_people[iDepart]['dist_to_open1'] < trial_values['diameter_doubled']):
                        closest_id = sorted_people[iDepart]['id']
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):0500:sorted_people[{iDepart}]['dist_to_open1'] = {sorted_people[iDepart]['dist_to_open1']}\n")
                            f.write(f"update_crowd({frame}):0550:closest_id = {closest_id}\n")
                        if (trial_values['run_type'] == 'training'):
                            labels[closest_id] = 1
                        else:
                            sorted_people[iDepart]['exit_actual'] = 1
                            animal_type = sorted_people[iDepart]['person_type'] + '-' + sorted_people[iDepart]['vision_animal']
                            if (sorted_people[iDepart]['exit_actual'] == sorted_people[iDepart]['exit_predicted']):
                                # Increment count for the civilian's animal_type in both dictionaries
                                trial_exit_match[animal_type] = trial_exit_match.get(animal_type, 0) + 1
                                experiment_exit_match[animal_type] = experiment_exit_match.get(animal_type, 0) + 1
                            else:
                                trial_exit_missed[animal_type] = trial_exit_missed.get(animal_type, 0) + 1
                                experiment_exit_missed[animal_type] = experiment_exit_missed.get(animal_type, 0) + 1
                            # Instead of incrementing separate dictionaries, append results to the DataFrame
                            new_row_results = {
                                'animal_type': animal_type,
                                'match': int(sorted_people[iDepart]['exit_actual'] == sorted_people[iDepart]['exit_predicted']),
                                'missed': int(sorted_people[iDepart]['exit_actual'] != sorted_people[iDepart]['exit_predicted'])
                                }
                            decision_tree_results_trial = pd.concat([decision_tree_results_trial, pd.DataFrame([new_row_results])], ignore_index=True)
                            decision_tree_results_experiment = pd.concat([decision_tree_results_experiment, pd.DataFrame([new_row_results])], ignore_index=True)
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):0600:df_trial.loc[{closest_id}] before update\n")
                            # Convert Series to string before writing
                            if closest_id in df_trial.index:  # Check if closest_id is a valid index
                                f.write(df_trial.loc[df_trial.index == closest_id].to_string())
                            else:
                                f.write(f"update_crowd({frame}):0650:Warning: closest_id {closest_id} not found in df_trial.\n")
                            f.write("\n")
                        # find the rows in df_trial where the index (which is now 'id') matches closest_id. This ensures that we are correctly targeting the desired row.
                        df_trial.loc[df_trial.index == closest_id, 'evacuation_time'] = frame
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):0700:df_trial.loc[{closest_id}] after update\n")
                            # Convert Series to string before writing
                            if closest_id in df_trial.index:  # Check if closest_id is a valid index
                                f.write(df_trial.loc[df_trial.index == closest_id].to_string())
                            else:
                                f.write(f"update_crowd({frame}):0750:Warning: closest_id {closest_id} not found in df_trial.\n")
                            f.write("\n")
                        # Append a single row to the DataFrame
                        # make a new row as a dictionary
                        # we potentially have a pandas Series if there are more than 1 rows in df_trial that match the condition df_trial.index == closest_id. so we need either .iloc[0] or .item()
                        if closest_id in df_trial.index:  # Check if closest_id is a valid index
                            new_row_experiment = {
                                'color': df_trial.loc[df_trial.index == closest_id, 'color'].iloc[0],
                                'evacuation_time': frame,
                                'id': trial_values['crowd_size'] + math.ceil(elapsed_time),
                                'genre': df_trial.loc[df_trial.index == closest_id, 'genre'].iloc[0],
                                'trial': trial_values['trial_number'],
                                'person_type': df_trial.loc[df_trial.index == closest_id, 'person_type'].iloc[0],
                                'vision_animal': df_trial.loc[df_trial.index == closest_id, 'vision_animal'].iloc[0]
                                }
                            # append the new row using the loc method
                            # append to len(df_experiment), which points to the next row
                            df_experiment.loc[len(df_experiment)] = new_row_experiment
                            with open(trial_values['filename_debug'], "a") as f:
                                # debug print [len(df_experiment)-1], which points to the last row
                                f.write(f"update_crowd({frame}):0900:df_experiment.loc[{len(df_experiment)-1}]\n")
                                # Convert Series to string before writing
                                f.write(df_experiment.loc[len(df_experiment)-1].to_string()) 
                                f.write("\n")
                        else:
                            with open(trial_values['filename_debug'], "a") as f:
                                f.write(f"update_crowd({frame}):0775:Warning: closest_id {closest_id} not found in df_trial.\n")
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write("update_crowd({frame}):0800:new_row_experiment\n")
                            json.dump(new_row_experiment, f)
                            f.write("\n")
                        # remove this individual from the simulation
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):1000:remove this individual from the simulation\n")
                            f.write(f"update_crowd({frame}):1100:before removal, len(crowd) = {len(crowd)}\n")
                        crowd = [person for person in crowd if person['id'] != closest_id]
                        removed = True
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):1200:after removal, len(crowd) = {len(crowd)}\n")
                            f.write(f"update_crowd({frame}):1300:closest_id = {closest_id}, removed = {removed}\n")
                iDepart += 1
                with open(trial_values['filename_debug'], "a") as f:
                    f.write(f"update_crowd({frame}):1400:removed = {removed}, iDepart = {iDepart}\n")
            # somebody just departed, fewer people remain now
            people_remaining = len(crowd)
        # is anybody still here
        if people_remaining > 0:
            sorted_people = sorted(crowd, key=lambda x:x['dist_to_open2'])
            with open(trial_values['filename_debug'], "a") as f:
                f.write(f"update_crowd({frame}):1500:len(sorted_people) = {len(sorted_people)}\n")
            iDepart = 0
            removed = False
            while ((iDepart < len(sorted_people)) and (removed == False)):
                with open(trial_values['filename_debug'], "a") as f:
                    f.write(f"update_crowd({frame}):1600:people_remaining = {people_remaining}\n")
                # who is closest to the open side of exit 2
                if (sorted_people[iDepart]['genre'] == 'civilian'):
                    with open(trial_values['filename_debug'], "a") as f:
                        f.write(f"update_crowd({frame}):1700:(sorted_people[{iDepart}]['genre'] == 'civilian')\n")
                    # are they close enough to depart
                    # (1 * diameter) causes people to get stuck
                    if (sorted_people[iDepart]['dist_to_open2'] < trial_values['diameter_doubled']):
                        closest_id = sorted_people[iDepart]['id']
                        if (trial_values['run_type'] == 'training'):
                            labels[closest_id] = 1
                        else:
                            sorted_people[iDepart]['exit_actual'] = 2
                            animal_type = sorted_people[iDepart]['person_type'] + '-' + sorted_people[iDepart]['vision_animal']
                            if (sorted_people[iDepart]['exit_actual'] == sorted_people[iDepart]['exit_predicted']):
                                # Increment count for the civilian's animal_type in both dictionaries
                                trial_exit_match[animal_type] = trial_exit_match.get(animal_type, 0) + 1
                                experiment_exit_match[animal_type] = experiment_exit_match.get(animal_type, 0) + 1
                            else:
                                trial_exit_missed[animal_type] = trial_exit_missed.get(animal_type, 0) + 1
                                experiment_exit_missed[animal_type] = experiment_exit_missed.get(animal_type, 0) + 1
                            # Instead of incrementing separate dictionaries, append results to the DataFrame
                            new_row_results = {
                                'animal_type': animal_type,
                                'match': int(sorted_people[iDepart]['exit_actual'] == sorted_people[iDepart]['exit_predicted']),
                                'missed': int(sorted_people[iDepart]['exit_actual'] != sorted_people[iDepart]['exit_predicted'])
                                }
                            decision_tree_results_trial = pd.concat([decision_tree_results_trial, pd.DataFrame([new_row_results])], ignore_index=True)
                            decision_tree_results_experiment = pd.concat([decision_tree_results_experiment, pd.DataFrame([new_row_results])], ignore_index=True)
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):1800:sorted_people[{iDepart}]['dist_to_open2'] = {sorted_people[iDepart]['dist_to_open2']}\n")
                        # use the "loc" method to access and modify
			# the value in the DataFrame
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):1900:df_trial.loc[{closest_id}] before update\n")
                            # Convert Series to string before writing
                            if closest_id in df_trial.index:  # Check if closest_id is a valid index
                                f.write(df_trial.loc[df_trial.index == closest_id].to_string()) 
                            else:
                                f.write(f"update_crowd({frame}):2000:Warning: closest_id {closest_id} not found in df_trial.\n")
                            f.write("\n")
                        # find the rows in df_trial where the index (which is now 'id') matches closest_id. This ensures that we are correctly targeting the desired row.
                        if closest_id in df_trial.index:  # Check if closest_id is a valid index
                            df_trial.loc[df_trial.index == closest_id, 'evacuation_time'] = frame
                            with open(trial_values['filename_debug'], "a") as f:
                                f.write(f"update_crowd({frame}):2100:df_trial.loc[{closest_id}] after update\n")
                                # Convert Series to string before writing
                                f.write(df_trial.loc[df_trial.index == closest_id].to_string()) 
                                f.write("\n")
                        else:
                            f.write(f"update_crowd({frame}):2200:Warning: closest_id {closest_id} not found in df_trial.\n")
                        # Append a single row to the DataFrame
                        # make a new row as a dictionary
                        # we potentially have a pandas Series if there are more than 1 rows in df_trial that match the condition df_trial.index == closest_id. so we need either .iloc[0] or .item()
                        if closest_id in df_trial.index:  # Check if closest_id is a valid index
                            new_row_experiment = {
                                'color': df_trial.loc[df_trial.index == closest_id, 'color'].iloc[0],
                                'evacuation_time': frame,
                                'id': trial_values['crowd_size'] + math.ceil(elapsed_time),
                                'genre': df_trial.loc[df_trial.index == closest_id, 'genre'].iloc[0],
                                'trial': trial_values['trial_number'],
                                'person_type': df_trial.loc[df_trial.index == closest_id, 'person_type'].iloc[0],
                                'vision_animal': df_trial.loc[df_trial.index == closest_id, 'vision_animal'].iloc[0]
                                }
                            # append the new row using the loc method
                            # append to len(df_experiment), which points to the next row
                            df_experiment.loc[len(df_experiment)] = new_row_experiment
                            if (trial_values['run_type'] == 'training'):
                                labels[closest_id] = 2
                        else:
                            with open(trial_values['filename_debug'], "a") as f:
                                f.write(f"update_crowd({frame}):2300:Warning: closest_id {closest_id} not found in df_trial.\n")
                        # remove this individual from the simulation
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write("remove this individual from the simulation\n")
                            f.write(f"update_crowd({frame}):2400:before removal, len(crowd) = {len(crowd)}\n")
                        crowd = [person for person in crowd if person['id'] != closest_id]
                        removed = True
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):2500:after removal, len(crowd) = {len(crowd)}\n")
                            f.write(f"update_crowd({frame}):2600:closest_id = {closest_id}, removed = {removed}\n")
                iDepart += 1
                with open(trial_values['filename_debug'], "a") as f:
                    f.write(f"update_crowd({frame}):2700:removed = {removed}, iDepart = {iDepart}\n")
            # somebody just departed, fewer people remain now
            people_remaining = len(crowd)

    # either:
    # instead of responders, now operate on firemen and medical
    # or:
    # initially populate responders like the crowd
    # then add responders to the crowd
    # Separate positions by genre, but also firemen vs. medical
    civilians = [person for person in crowd if person['genre'] == 'civilian']
    firemen = [person for person in crowd if person['person_type'] == 'fireman']
    emeMedTec = [person for person in crowd if person['person_type'] == 'medical']

    # only update the scatter plot when there are people remaining
    if len(civilians) > 0:
        scatter_civilians.set_offsets(np.c_[[person['position'] for person in civilians]])
    else:
        # Pass a 2D empty array to set_offsets
        scatter_civilians.set_offsets(np.empty((0, 2))) 
    if len(firemen) > 0:
        scatter_firemen.set_offsets(np.c_[[person['position'] for person in firemen]])
    else:
        # Pass a 2D empty array to set_offsets
        scatter_firemen.set_offsets(np.empty((0, 2))) 
    if len(emeMedTec) > 0:
        scatter_emeMedTec.set_offsets(np.c_[[person['position'] for person in emeMedTec]])
    else:
        # Pass a 2D empty array to set_offsets
        scatter_emeMedTec.set_offsets(np.empty((0, 2))) 

    # ... (Timer and occupancy display logic, adapted for dictionary access)
    # Update the timer and occupancy display
    if alarm_values['alarm_triggered']:
        # the longer the elapsed time, the more first responders arrive
        random_response = random.random()
        if (random_response * elapsed_time) > response_time_float:
            # a first responder does arrive during this frame
            exit_choice = random.randint(1, 2)
            if (exit_choice == 1):
                # emerge from exit_1
                dist_to_open1 = 0
                dist_to_open2 = half_diagonal
                responder_x = trial_values['open1_x']
                responder_y = trial_values['open1_y']
            else:
                # emerge from exit_2
                dist_to_open1 = half_diagonal
                dist_to_open2 = 0
                responder_x = trial_values['open2_x']
                responder_y = trial_values['open2_y']
            # is the responder a fireman or an emergency medical technician
            random_type = random.random()
            if (random_type < 0.5):
                # fireman
                responder_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'fireman', human_values))
            else:
                # emergency medical technician
                responder_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'medical', human_values))
            responder = {
                'angle': np.random.rand() * 2 * np.pi,
                'color': responder_values[0]['color'],
                # give each person an id
                'id': trial_values['crowd_size'] + math.ceil(elapsed_time), 
                'dist_to_open1': dist_to_open1,
                'dist_to_open2': dist_to_open2,
                'dist_to_civilian': trial_values['room_diagonal'],
                'dist_to_responder': trial_values['room_diagonal'],
                'entry_time': frame,
                'genre': responder_values[0]['genre'],
                'person_type': responder_values[0]['person_type'],
                'position': [responder_x, responder_y],
                'velocity_animal': responder_values[0]['velocity_animal'],
                'velocity': np.random.rand() * responder_values[0]['velocity_initial'],
                'velocity_initial': responder_values[0]['velocity_initial'],
                'velocity_limit': responder_values[0]['velocity_limit'],
                'vision_animal': responder_values[0]['vision_animal'],
                'vision_factor': responder_values[0]['vision_factor'],
                'vision_limit': responder_values[0]['vision_limit'],
                }
            crowd.append(responder)
            # Append a single row to the DataFrame
            # make a new row as a dictionary
            new_row_trial = {
                'color': responder_values[0]['color'],
                'evacuation_time': 0,
                'id': trial_values['crowd_size'] + math.ceil(elapsed_time),
                'genre': responder_values[0]['genre'],
                'person_type': responder_values[0]['person_type'],
                'vision_animal': responder_values[0]['vision_animal'],
                }
            # use .loc with the next available index to append the new row
            df_trial.loc[df_trial.index.max() + 1 if len(df_trial) > 0 else 0] = new_row_trial
        genre_counts = {}
        for somebody in crowd:
            genre = somebody['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        # Use the get method with a default value of 0 to avoid KeyError
        # in the initial frames of the animation after the alarm is triggered.
        # At this point, only civilians have been added to the crowd list,
        # so when the code iterates through crowd to populate genre_counts,
        # it only encounters the 'civilian' genre. Consequently, the
        # genre_counts dictionary only contains the 'civilian' key
        # and not 'responder'.
        present_civilians = genre_counts.get('civilian', 0)
        present_people = len(crowd)
        present_responders = genre_counts.get('responder', 0)
        responders_to_remove = present_responders - present_civilians
        # Check if we need to remove any responders
        if responders_to_remove > 0:  
            # improve: check both exits,
            # could remove a responder who is far from either exit
            # send responders to an exit
            sorted_people = sorted(crowd, key=lambda x: x['dist_to_open2'])
            iDepart = 0
            while responders_to_remove > 0 and iDepart < len(sorted_people):
                if sorted_people[iDepart]['genre'] == 'responder':
                    closest_id = sorted_people[iDepart]['id']
                    # Find the row with matching 'id'
                    # in df_trial
                    # use the "loc" method to access and modify
                    # the value in the DataFrame
                    # find the rows in df_trial where the index (which is now 'id') matches closest_id. This ensures that we are correctly targeting the desired row.
                    if closest_id in df_trial.index:  # Check if closest_id is a valid index
                        df_trial.loc[df_trial.index == closest_id, 'evacuation_time'] = frame
                        # Append a single row to the DataFrame
                        # make a new row as a dictionary
                        # we potentially have a pandas Series if there are more than 1 rows in df_trial that match the condition df_trial.index == closest_id. so we need either .iloc[0] or .item()
                        new_row_experiment = {
                            'color': df_trial.loc[df_trial.index == closest_id, 'color'].iloc[0],
                            'evacuation_time': frame,
                            'id': trial_values['crowd_size'] + math.ceil(elapsed_time),
                            'genre': df_trial.loc[df_trial.index == closest_id, 'genre'].iloc[0],
                            'trial': trial_values['trial_number'],
                            'person_type': df_trial.loc[df_trial.index == closest_id, 'person_type'].iloc[0],
                            'vision_animal': df_trial.loc[df_trial.index == closest_id, 'vision_animal'].iloc[0]
                            }
                        # append the new row using the loc method
                        df_experiment.loc[len(df_experiment)] = new_row_experiment
                    else:
                        with open(trial_values['filename_debug'], "a") as f:
                            f.write(f"update_crowd({frame}):Warning: closest_id {closest_id} not found in df_trial.\n")
                    crowd = [person for person in crowd if person['id'] != closest_id]
                    responders_to_remove -= 1
                iDepart += 1
        text_object.set_text(f"Elapsed time: {elapsed_time / 2:.1f} Civilians: {present_civilians} Responders: {present_responders}")

    return scatter_civilians, scatter_firemen, scatter_emeMedTec, text_object

def run_simulation():
    global alarm_values, crowd, decision_tree_results_experiment, decision_tree_results_trial, df_trial, experiment_exit_match, experiment_exit_missed, features, human_values, labels, scatter_civilians, scatter_firemen, scatter_emeMedTec, text_object, trial_exit_match, trial_exit_missed, trial_values

    # Initialize a dictionary to store counts for the current run
    trial_exit_match = {
        'able-giraffe': 0,
        'able-chameleon': 0,
        'challenged-giraffe': 0,
        'challenged-chameleon': 0
    }
    trial_exit_missed = {
        'able-giraffe': 0,
        'able-chameleon': 0,
        'challenged-giraffe': 0,
        'challenged-chameleon': 0
    }

    # Initialize an empty DataFrame to store decision tree results
    decision_tree_results_trial = pd.DataFrame(columns=['animal_type', 'match', 'missed'])

    # initialization that needs to happen again for each simulation
    # Set up the figure and axes with realistic scaling
    fig, ax = plt.subplots()
    ax.set_xlim(0, trial_values['room_length'])
    ax.set_ylim(0, trial_values['room_width'])

    ax.set_aspect('equal')  # Ensure correct aspect ratio

    # Place exits randomly while obeying the half-diagonal rule
    # and keeping exits away from the walls
    while True:
        # exit 1 location
        exit1_x = np.random.rand() * (trial_values['room_length'] - trial_values['exit_size']) + trial_values['half_exit_size']
        exit1_y = np.random.rand() * (trial_values['room_width'] - trial_values['exit_size']) + trial_values['half_exit_size']
        # distance from exit1 to the closest of (the left and right walls)
        exit1_x_wall = min(exit1_x - 0, trial_values['room_length'] - exit1_x)
        # distance from exit1 to the closest of (the top and bottom walls)
        exit1_y_wall = min(exit1_y - 0, trial_values['room_width'] - exit1_y)
        # exit 2 location
        exit2_x = np.random.rand() * (trial_values['room_length'] - trial_values['exit_size']) + trial_values['half_exit_size']
        exit2_y = np.random.rand() * (trial_values['room_width'] - trial_values['exit_size']) + trial_values['half_exit_size']
        # distance from exit2 to the closest of (the left and right walls)
        exit2_x_wall = min(exit2_x - 0, trial_values['room_length'] - exit2_x)
        # distance from exit2 to the closest of (the top and bottom walls)
        exit2_y_wall = min(exit2_y - 0, trial_values['room_width'] - exit2_y)
        if (distance(exit1_x, exit1_y, exit2_x, exit2_y) >= half_diagonal) and (exit1_x_wall > trial_values['exit_size']) and (exit1_y_wall > trial_values['exit_size']) and (exit2_x_wall > trial_values['exit_size']) and (exit2_y_wall > trial_values['exit_size']):
            break

    # Assign open sides to exits (0: top, 1: right, 2: bottom, 3: left)
    exit1_open_side = np.random.randint(4)
    exit2_open_side = np.random.randint(4)

    # Draw the exits
    draw_exit(exit1_x, exit1_y, trial_values['exit_size'], exit1_open_side)
    draw_exit(exit2_x, exit2_y, trial_values['exit_size'], exit2_open_side)

    # Locate the middle of the open side of each exit
    match exit1_open_side:
        case 0:
            # the top is open
            trial_values['open1_x'] = exit1_x
            trial_values['open1_y'] = exit1_y + trial_values['half_exit_size']
        case 1:
            # the right is open
            trial_values['open1_x'] = exit1_x + trial_values['half_exit_size']
            trial_values['open1_y'] = exit1_y
        case 2:
            # the bottom is open
            trial_values['open1_x'] = exit1_x
            trial_values['open1_y'] = exit1_y - trial_values['half_exit_size']
        case 3:
            # the left is open
            trial_values['open1_x'] = exit1_x - trial_values['half_exit_size']
            trial_values['open1_y'] = exit1_y
    match exit2_open_side:
        case 0:
            # the top is open
            trial_values['open2_x'] = exit2_x
            trial_values['open2_y'] = exit2_y + trial_values['half_exit_size']
        case 1:
            # the right is open
            trial_values['open2_x'] = exit2_x + trial_values['half_exit_size']
            trial_values['open2_y'] = exit2_y
        case 2:
            # the bottom is open
            trial_values['open2_x'] = exit2_x
            trial_values['open2_y'] = exit2_y - trial_values['half_exit_size']
        case 3:
            # the left is open
            trial_values['open2_x'] = exit2_x - trial_values['half_exit_size']
            trial_values['open2_y'] = exit2_y

    crowd = []
    features = []
    labels = []
    for i in range(trial_values['crowd_size']):
        random_velocity = random.random() 
        # 80% good velocity, 20% bad velocity
        random_vision = random.random()
        if trial_values['debug']:
            # begin print 20250110 debug
            print(f"run_simulation: i = {i:3}, random_velocity = {random_velocity}, random_vision = {random_vision}")
            # end print 20250110 debug
        # 80% good vision, 20% bad vision
        if (random_velocity < 0.8) and (random_vision < 0.8):  
            individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'able' and item['vision_animal'] == 'giraffe', human_values))
        elif (random_velocity < 0.8) and (random_vision >= 0.8):  
            individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'able' and item['vision_animal'] == 'chameleon', human_values))
        elif (random_velocity >= 0.8) and (random_vision < 0.8):  
            individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'challenged' and item['vision_animal'] == 'giraffe', human_values))
        else:
            individual_values = list(filter(lambda item: item['genre'] == 'civilian' and item['person_type'] == 'challenged' and item['vision_animal'] == 'chameleon', human_values))
        if trial_values['debug']:
            # begin print 20250110 debug
            # Print the individual values list
            print("run_simulation: 20250110 debug individual_values")
            pprint.pprint(individual_values)  # Print the individual values list
            # end print 20250110 debug
        initial_x = np.random.rand() * (trial_values['room_length'] - trial_values['diameter']) + trial_values['radius']
        initial_y = np.random.rand() * (trial_values['room_width'] - trial_values['diameter']) + trial_values['radius']
        dist_to_open1 = distance(initial_x, initial_y, trial_values['open1_x'], trial_values['open1_y'])
        dist_to_open2 = distance(initial_x, initial_y, trial_values['open2_x'], trial_values['open2_y'])
        crowd.append({
            'angle': np.random.rand() * 2 * np.pi,
            'color': individual_values[0]['color'],
            'id': i, # give each person an id
            'dist_to_open1': dist_to_open1,
            'dist_to_open2': dist_to_open2,
            'dist_to_civilian': trial_values['room_diagonal'],
            'dist_to_responder': trial_values['room_diagonal'],
            'entry_time': 0,
            # initialize everybody to never exit
            # most people will exit via either 1 or 2
            'exit_actual': 0,
            'exit_predicted': 0,
            'genre': individual_values[0]['genre'],
            'person_type': individual_values[0]['person_type'],
            'position': [initial_x, initial_y],
            'velocity_animal': individual_values[0]['velocity_animal'],
            'velocity': (np.random.rand() * individual_values[0]['velocity_initial']),
            'velocity_initial': individual_values[0]['velocity_initial'],
            'velocity_limit': individual_values[0]['velocity_limit'],
            'vision_animal': individual_values[0]['vision_animal'],
            'vision_factor': individual_values[0]['vision_factor'],
            'vision_limit': half_diagonal * individual_values[0]['vision_factor']
        })
        features.append({
            'position_x': initial_x,
            'position_y': initial_y,
            'velocity': crowd[i]['velocity'], 
            'vision_limit': crowd[i]['vision_limit']
        })
        # initialize everybody to never exit
        # most people will exit via either 1 or 2
        labels.append(0)

    # will use this to record when each civilian departs or is removed
    # Initialize an empty DataFrame
    df_trial = pd.DataFrame(columns=['color', 'evacuation_time', 'id', 'genre', 'person_type', 'vision_animal'])
    for i in range(trial_values['crowd_size']):
        # Append a single row to the DataFrame
        # make a new row as a dictionary
        new_row_trial = {
            'color': crowd[i]['color'],
            'evacuation_time': 0,
            'id': i, # give each person an id
            'genre': crowd[i]['genre'],
            'person_type': crowd[i]['person_type'],
            'vision_animal': crowd[i]['vision_animal']
            }
        # use .loc with the next available index to append the new row
        df_trial.loc[df_trial.index.max() + 1 if len(df_trial) > 0 else 0] = new_row_trial
    # To set the index of a Pandas DataFrame to a unique ID,
    # we can use the set_index() method
    # we have a column with unique values that we want to use as the index
    df_trial.set_index('id', inplace=True)
    if trial_values['debug']:
        # begin 20250112 debug
        print("run_simulation(): Initialize a DataFrame: df_trial")
        print(df_trial)
        # begin 20250112 debug

    # Initialize the scatter plot with individual colors and markers
    scatter_civilians = ax.scatter(
        [person['position'][0] for person in crowd],
        [person['position'][1] for person in crowd],
        color=[person['color'] for person in crowd],
        marker='h',  # civilians are hexagons
        s=trial_values['diameter']
       )

    individual_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'fireman', human_values))
    scatter_firemen = ax.scatter(
        [],
        [],
        color= individual_values[0]['color'],
        marker='*',
        s=trial_values['diameter']
        )

    individual_values = list(filter(lambda item: item['genre'] == 'responder' and item['person_type'] == 'medical', human_values))
    scatter_emeMedTec = ax.scatter(
        [],
        [],
        color= individual_values[0]['color'],
        marker='+',
        s=trial_values['diameter']
        )

    alarm_values = {
        # Alarm rings sometime within the first alarm_latest frames
        'alarm_time': np.random.randint(trial_values['alarm_latest']),
        # Alarm trigger
        'alarm_triggered': False
        }
        
    # Initialize the text object for the timer and occupancy
    text_object = ax.text(
        trial_values['announce_x'],
        trial_values['announce_y'],
        "",
        color=trial_values['announce_color'],
        fontsize=trial_values['announce_fontsize'],
        ha=trial_values['announce_alignment']
        )

    # Create the animation using FuncAnimation
    ani = animation.FuncAnimation(
        fig,
        update_crowd,
        frames=trial_values['frame_duration'],
        interval=trial_values['frame_interval'],
        blit=True
        )

    # Animation Storage: store the FuncAnimation objects generated
    # in each simulation run.
    # Appending the animation object 'ani' instead of the module 'animation'
    list_animations.append(ani)

    # Save the animation as an MP4 file
    # The easiest way to find your saved file is to look in the "Files" tab
    # in the left sidebar of your Colab environment.
    # Files saved in your Colab environment are not permanent.
    # If your Colab runtime disconnects, you'll lose any saved files.
    # To keep your animation, make sure to download 'filename.mp4'
    # to your local machine before your Colab session ends. You can do this
    # by right-clicking on the file in the "Files" tab and selecting "Download".
    if (trial_values['run_type'] == 'simulation'):
        ani_file_name = f"20250115l{trial_values['trial_number']:0{trial_values['trial_width']}}wcp.mp4"
    else:
        ani_file_name = f"train115l{trial_values['train_number']:0{trial_values['train_width']}}wcp.mp4"
    ani.save(ani_file_name, writer='ffmpeg')
    zip_file.write(ani_file_name)

    if (trial_values['run_type'] == 'simulation'):
        filename_trial = f"20250115l{trial_values['trial_number']:0{trial_values['trial_width']}}wcp.txt"
    else:
        filename_trial = f"train115l{trial_values['train_number']:0{trial_values['train_width']}}wcp.txt"
    with open(filename_trial, "w") as f:
        f.write("Summary of all people\n")
        # Pandas Series or DataFrame objects, which cannot be
        # directly written to a file using f.write. Instead, you need
        # to convert them to strings before writing.
        # Convert describe output to string before writing
        f.write(str(df_trial[['evacuation_time']].describe()))  
        f.write("\n")
        f.write("Summary of all combinations\n")
        # Convert groupby and describe output to string before writing
        f.write(str(df_trial.groupby(['genre', 'person_type', 'vision_animal'])['evacuation_time'].describe()))  
        f.write("\n")
        f.write(f"Trial {trial_values['trial_number']:0{trial_values['trial_width']}} Decision Tree Results\n")
        f.write("trial exit match\n")
        json.dump(trial_exit_match, f)
        f.write("\n")
        f.write("trial exit missed\n")
        json.dump(trial_exit_missed, f)
        f.write("\n")
        # Calculate and print statistics
        if (trial_values['run_type'] == 'simulation'):
            print(f"Trial {trial_values['trial_number']:0{trial_values['trial_width']}} Decision Tree Results")
            print(decision_tree_results_trial.groupby('animal_type').agg(['sum', 'mean', 'std']))
            # ... (write to file)
            f.write(f"Trial {trial_values['trial_number']:0{trial_values['trial_width']}} Decision Tree Results\n")
            f.write(str(decision_tree_results_trial.groupby('animal_type').agg(['sum', 'mean', 'std'])))
            f.write("\n")
    zip_file.write(filename_trial)
    if (trial_values['run_type'] == 'simulation'):
        df_xlsx_name = f"20250115l{trial_values['trial_number']:0{trial_values['trial_width']}}wcp.xlsx"
    else:
        df_xlsx_name = f"train115l{trial_values['train_number']:0{trial_values['train_width']}}wcp.xlsx"
    # avoid writing the index to the excel file
    df_trial.to_excel(df_xlsx_name, index=False) #Added index=False
    zip_file.write(df_xlsx_name)

    # Generate Box-and-Whisker Plots:
    plt.figure()  # Create a new figure for the box plot
    # Create a new column combining person_type and vision_animal
    df_trial['animal_type'] = df_trial['person_type'] + '-' + df_trial['vision_animal']
    # begin evac debug
    if trial_values['debug']:
        print("run_simulation(): evac debug df_trial")
        print(df_trial)  # Print the entire DataFrame
        # Print unique animal types
        print("run_simulation(): evac debug df_trial['animal_type'].unique()")  
        print(df_trial['animal_type'].unique())  # Print unique animal types
    # end evac debug

    # Group data by animal type and get evacuation times
    # civilians have both speed and vision variable, 4 combinations
    # responders have speed and vision fixed, 2 combinations
    animal_types = ['able-giraffe', 'able-chameleon', 'challenged-giraffe', 'challenged-chameleon', 'fireman-eagle', 'medical-owl']
    evacuation_times = [df_trial[df_trial['animal_type'] == animal_type]['evacuation_time'] for animal_type in animal_types]

    plt.boxplot(evacuation_times)
    # Use animal_types for x-axis labels
    # rotate to avoid overwrite
    plt.xticks([1, 2, 3, 4, 5, 6], animal_types, rotation=45)  
    plt.ylabel('Evacuation Time')
    if (trial_values['run_type'] == 'simulation'):
        plt.title(f"Trial {trial_values['trial_number']:0{trial_values['trial_width']}} Distribution of Evacuation Times by Person Type")
        box_file_name = f"20250115l{trial_values['trial_number']:0{trial_values['trial_width']}}box.png"
    else:
        plt.title(f"Train {trial_values['train_number']:0{trial_values['train_width']}} Distribution of Evacuation Times by Person Type")
        box_file_name = f"train115l{trial_values['train_number']:0{trial_values['train_width']}}box.png"
    plt.savefig(box_file_name)
    zip_file.write(box_file_name)
    if (trial_values['run_type'] == 'simulation'):
        # Generate dual bar chart
        summary = decision_tree_results_trial.groupby('animal_type').agg('sum') 
        bar_chart_types = summary.index
        matches = summary['match']
        misses = summary['missed']
        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        index = range(len(bar_chart_types))

        fig, ax = plt.subplots()
        rects1 = ax.bar(index, matches, bar_width, label='Match')
        # Shift the 'missed' bars slightly to the right
        rects2 = ax.bar([i + bar_width for i in index], misses, bar_width, label='Missed')

        # Set the labels, title, and legend
        ax.set_xlabel('Civilian Type')
        ax.set_ylabel('Count')
        ax.set_title(f"Trial {trial_values['trial_number']:0{trial_values['trial_width']}} Decision Tree Results")
        bar_file_name = f"20250115l{trial_values['trial_number']:0{trial_values['trial_width']}}bar.png"
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(bar_chart_types, rotation=45)  # Rotate labels if needed
        ax.legend()

        # Display the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig(bar_file_name)
        zip_file.write(bar_file_name)

# main program starts here
# matplotlib: With Interactive Mode On: Plots are displayed
# as soon as they are created
# we don't want this because we might have many trials
plt.ioff()  # Turn interactive mode off
# initialization that only needs to happen once for all simulations
# read this in from the data file
room_length_inches = int(room_length) * 12  # 100 feet in inches
# read this in from the data file
room_width_inches = int(room_width) * 12  # 20 feet in inches
# Pythagorean Theorem
room_diagonal = (room_length_inches**2 + room_width_inches**2)**0.5  
# Add exits with realistic size, orientation, and placement
# read this in from the data file
exit_size_inches = int(exit_size)  # Side length of the square exit in inches
half_diagonal = room_diagonal / 2  # Half-diagonal of the room
# Crowd Initialization with Genres and Categories
# Access colors from the CSS4 colors dictionary
# Headline = sienna
sienna = mcolors.CSS4_COLORS['sienna']
# Exit Interiors = green
green = mcolors.CSS4_COLORS['green']
# Exit Walls = orange
orange = mcolors.CSS4_COLORS['orange']
# High velocity fast civilians are able
# Civilians Able = navy
navy = mcolors.CSS4_COLORS['navy']
civilian_able_velocity_animal = 'springbok'
# read this in from the data file, then convert to inches per second
civilian_able_velocity_initial_inches = float(civilian_able_velocity_initial) * 12 
# read this in from the data file, then convert to inches per second
civilian_able_velocity_limit_inches = float(civilian_able_velocity_limit) * 12
# Long vision keen sighted civilians are keen
civilian_keen_vision_animal = 'giraffe'
# read this in from the data file, then convert from string to float
civilian_keen_vision_factor_float = float(civilian_keen_vision_factor)
civilian_keen_vision_limit = civilian_keen_vision_factor_float * half_diagonal
# Low velocity speed challenged civilians are challenged
# Civilians Challenged = lightskyblue
lightskyblue = mcolors.CSS4_COLORS['lightskyblue']
civilian_challenged_velocity_animal = 'kangaroo'
# read this in from the data file, then convert to inches per second
civilian_challenged_velocity_initial_inches = float(civilian_challenged_velocity_initial) * 12
# read this in from the data file, then convert to inches per second
civilian_challenged_velocity_limit_inches = float(civilian_challenged_velocity_limit) * 12
# Short vision challenged civilians are foggy
civilian_foggy_vision_animal = 'chameleon'
# read this in from the data file, then convert from string to float
civilian_foggy_vision_factor_float = float(civilian_foggy_vision_factor)
civilian_foggy_vision_limit = civilian_foggy_vision_factor_float * half_diagonal
# Responders Firemen = crimson
crimson = mcolors.CSS4_COLORS['crimson']
responder_fireman_velocity_animal = 'cheetah'
# read this in from the data file, then convert to inches per second
responder_fireman_velocity_initial_inches = float(responder_fireman_velocity_initial) * 12
# read this in from the data file, then convert to inches per second
responder_fireman_velocity_limit_inches = float(responder_fireman_velocity_limit) * 12
responder_fireman_vision_animal = 'eagle'
# read this in from the data file, then convert from string to float
responder_fireman_vision_factor_float = float(responder_fireman_vision_factor)
responder_fireman_vision_limit = responder_fireman_vision_factor_float * half_diagonal
# Responders Medical = lightcoral
lightcoral = mcolors.CSS4_COLORS['lightcoral']
responder_medical_velocity_animal = 'lion'
# read this in from the data file, then convert to inches per second
responder_medical_velocity_initial_inches = float(responder_medical_velocity_initial) * 12
# read this in from the data file, then convert to inches per second
responder_medical_velocity_limit_inches = float(responder_medical_velocity_limit) * 12
responder_medical_vision_animal = 'owl'
# read this in from the data file, then convert from string to float
responder_medical_vision_factor_float = float(responder_medical_vision_factor)
responder_medical_vision_limit = responder_medical_vision_factor_float * half_diagonal

# read this in from the data file, then convert from string to int
diameter_int = int(diameter)
two_diameter = 2 * diameter_int
radius = diameter_int / 2
# read this in from the data file
# tune this to bring in first responders
response_time_float = float(response_time)

# vision_factor of 0.600 still everybody sees the exits all the time
# vision_factor of 0.300 works with np.clip, but fails with wall bouncing
# vision_factor of 0.150 everybody eventually exits, but there is some wandering
# vision_factor of 0.075 everybody eventually exits, plus comical wandering

human_values = [
    # civilian fast velocity and keen vision
    {
        'color': navy,
        'genre': 'civilian',
        'person_type': 'able',
        'velocity_animal': civilian_able_velocity_animal,
        'velocity_initial': civilian_able_velocity_initial_inches,
        'velocity_limit': civilian_able_velocity_limit_inches,
        'vision_animal': civilian_keen_vision_animal,
        'vision_factor': civilian_keen_vision_factor_float,
        'vision_limit': civilian_keen_vision_limit
    },
    # civilian fast velocity and foggy vision
    {
        'color': navy,
        'genre': 'civilian',
        'person_type': 'able',
        'velocity_animal': civilian_able_velocity_animal,
        'velocity_initial': civilian_able_velocity_initial_inches,
        'velocity_limit': civilian_able_velocity_limit_inches,
        'vision_animal': civilian_foggy_vision_animal,
        'vision_factor': civilian_foggy_vision_factor_float,
        'vision_limit': civilian_foggy_vision_limit
    },
    # civilian velocity challenged and keen vision
    {
        'color': lightskyblue,
        'genre': 'civilian',
        'person_type': 'challenged',
        'velocity_animal': civilian_challenged_velocity_animal,
        'velocity_initial': civilian_challenged_velocity_initial_inches,
        'velocity_limit': civilian_challenged_velocity_limit_inches,
        'vision_animal': civilian_keen_vision_animal,
        'vision_factor': civilian_keen_vision_factor_float,
        'vision_limit': civilian_keen_vision_limit
    },
    # civilian velocity challenged and foggy vision
    {
        'color': lightskyblue,
        'genre': 'civilian',
        'person_type': 'challenged',
        'velocity_animal': civilian_challenged_velocity_animal,
        'velocity_initial': civilian_challenged_velocity_initial_inches,
        'velocity_limit': civilian_challenged_velocity_limit_inches,
        'vision_animal': civilian_foggy_vision_animal,
        'vision_factor': civilian_foggy_vision_factor_float,
        'vision_limit': civilian_foggy_vision_limit
    },
    {
        'color': crimson,
        'genre': 'responder',
        'person_type': 'fireman',
        'velocity_animal': responder_fireman_velocity_animal,
        'velocity_initial': responder_fireman_velocity_initial_inches,
        'velocity_limit': responder_fireman_velocity_limit_inches,
        'vision_animal': responder_fireman_vision_animal,
        'vision_factor': responder_fireman_vision_factor_float,
        'vision_limit': responder_fireman_vision_limit
    },
    {
        'color': lightcoral,
        'genre': 'responder',
        'person_type': 'medical',
        'velocity_animal': responder_medical_velocity_animal,
        'velocity_initial': responder_medical_velocity_initial_inches,
        'velocity_limit': responder_medical_velocity_limit_inches,
        'vision_animal': responder_medical_vision_animal,
        'vision_factor': responder_medical_vision_factor_float,
        'vision_limit': responder_medical_vision_limit
    }
    ]
        
# Set up duration and timing
# reading crowd_size from the csv made it a float, if not a string
crowd_size_int = int(crowd_size)
# read this in from the data file
frame_duration_int = int(frame_duration)
# up to 2 people depart during each frame
# worst case everybody waits in line by a single exit
# so up to 1 person departs during each frame
# this is the latest alarm that still lets everybody depart
alarm_latest = frame_duration_int - crowd_size_int
# each responder emerges from an exit
# and travels to a corner to get out of the way
# before doing anything else
# read this in from the data file, then convert from string to int
responder_deployment_int = int(responder_deployment)

# read this in from the data file, then convert from string to boolean
evac_debug_bool = string_to_bool(evac_debug)
file_debug_name = "20250115l-debug.txt"
# will hold all output files
zip_file = zipfile.ZipFile('20250115l.zip', 'w')
# Open the file in write mode to start fresh
with open(file_debug_name, "w") as f:
    f.write("20250115l-debug\n")
# read this in from the data file, then convert from string to int
max_train_int = int(max_train)
# read this in from the data file, then convert from string to int
train_width_int = int(train_width)
# read this in from the data file, then convert from string to int
max_trials_int = int(max_trials)
# read this in from the data file, then convert from string to int
trial_width_int = int(trial_width)

if evac_debug_bool:
    # begin print 20250110 debug
    print("main: 20250110 debug human_values")  # Print the human values list
    pprint.pprint(human_values)  # Print the human values list
    # end print 20250110 debug

# Initialize the text object for the timer and occupancy
half_room_length_inches = room_length_inches / 2
nine_room_width_inches = 0.9 * room_width_inches

# Animation Storage: Create a list
# to store the FuncAnimation objects generated
# in each simulation run.
list_animations = []

# Data Accumulation: Make a DataFrame to store the df_trial
# from each simulation run.
# initialize an empty DataFrame and append each row to it as we generate them.
# This approach could be less efficient but avoids the need for concatenation.
# Initialize an empty DataFrame:
df_experiment = pd.DataFrame(columns=['color', 'evacuation_time', 'id', 'genre', 'person_type', 'trial', 'vision_animal'])

# Declare/Define a Dictionary: Using curly braces.
trial_values = {
    'alarm_latest': alarm_latest,
    'announce_alignment': 'center',
    'announce_color': 'sienna',
    'announce_fontsize': 14,
    'announce_x': half_room_length_inches,
    'announce_y': nine_room_width_inches,
    'debug': evac_debug_bool,
    'crowd_size': crowd_size_int,
    'diameter': diameter_int,
    'diameter_doubled': two_diameter,
    'exit_size': exit_size_inches,
    'filename_debug': file_debug_name,
    'frame_duration': frame_duration_int,
    'frame_interval': 50,
    'half_exit_size': exit_size_inches / 2,
    'open1_x': 0.0,
    'open1_y': 0.0,
    'open2_x': 0.0,
    'open2_y': 0.0,
    'radius': radius,
    'room_diagonal': room_diagonal,
    'room_length': room_length_inches,
    'room_width': room_width_inches,
    'run_type': 'training',
    'train_number': 0,
    'train_width': train_width_int,
    'trial_number': 0,
    'trial_width': trial_width_int
    }

# Initialize a dictionary to store counts for the entire experiment
experiment_exit_match = {
    'able-giraffe': 0,
    'able-chameleon': 0,
    'challenged-giraffe': 0,
    'challenged-chameleon': 0
}

experiment_exit_missed = {
    'able-giraffe': 0,
    'able-chameleon': 0,
    'challenged-giraffe': 0,
    'challenged-chameleon': 0
}

# Initialize an empty DataFrame to store decision tree results
decision_tree_results_experiment = pd.DataFrame(columns=['animal_type', 'match', 'missed'])

for iter_train in range(max_train_int):
    trial_values['train_number'] = iter_train
    with open(trial_values['filename_debug'], "a") as f:
        f.write(f"main(): training run = {iter_train}\n")
    run_simulation()

# Decision Tree Training
# Use a list comprehension to extract the numerical features from the dictionaries in features
numerical_features = [[feature['position_x'], feature['position_y'], feature['velocity'], feature['vision_limit']] for feature in features]

X_train, X_test, y_train, y_test = train_test_split(
    numerical_features, labels, test_size=0.25, random_state=42
)

# Ensure X_train is a numerical array
X_train = np.array(X_train)

# Ensure y_train is a numerical array
y_train = np.array(y_train)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

trial_values['run_type'] = 'simulation'
for iter_trial in range(max_trials_int):
    # Access Dictionary Element: Using square brackets with the key.
    trial_values['trial_number'] = iter_trial
    with open(trial_values['filename_debug'], "a") as f:
        f.write(f"main(): trial = {iter_trial}\n")
    run_simulation()

# Random Choice: After the simulation loop, use random.choice(animations)
# to select one animation randomly.
# Display the animation in Google Colab using HTML
# this has never worked, has never put an animation in colab
# instead, it displays the final frame
# HTML(random.choice(list_animations).to_jshtml())

# Use the matplotlib.pyplot.show() function after displaying the animation
# and again after displaying the box plot to force both to appear
# in the same cell's output.
# this has never worked, has never displayed only one single trial in colab
# instead, it displays every trial
# plt.show()

# Calculate and Print Numerical Statistics:
# display to cell
print("Experiment Summary of all people")
print(df_experiment[['evacuation_time']].describe())
print("Experiment Summary of all combinations")
print(df_experiment.groupby(['genre', 'person_type', 'vision_animal'])['evacuation_time'].describe())
# write to file
df_experiment_name = "20250115l-wcp.txt"
with open(df_experiment_name, "w") as f:
    f.write("Experiment Summary of all people\n")
    f.write(str(df_experiment[['evacuation_time']].describe()))
    f.write("\n")
    f.write("Experiment Summary of all combinations\n")
    f.write(str(df_experiment.groupby(['genre', 'person_type', 'vision_animal'])['evacuation_time'].describe()))  
    f.write("\n")
    f.write("Experiment Decision Tree Results\n")
    f.write("experiment exit match\n")
    json.dump(experiment_exit_match, f)
    f.write("\n")
    f.write("experiment exit missed\n")
    json.dump(experiment_exit_missed, f)
    f.write("\n")
    # Calculate and print statistics
    print("Experiment Decision Tree Results")
    print(decision_tree_results_experiment.groupby('animal_type').agg(['sum', 'mean', 'std']))
    # ... (write to file)
    f.write("Experiment Decision Tree Results\n")
    f.write(str(decision_tree_results_experiment.groupby('animal_type').agg(['sum', 'mean', 'std'])))
    f.write("\n")
zip_file.write(df_experiment_name)
df_xlsx_name = "20250115l-wcp.xlsx"
df_experiment.to_excel(df_xlsx_name, index=False) #Added index=False
zip_file.write(df_xlsx_name)

# Generate Box-and-Whisker Plots:
plt.figure()  # Create a new figure for the box plot
# Create a new column combining person_type and vision_animal
df_experiment['animal_type'] = df_experiment['person_type'] + '-' + df_experiment['vision_animal']
# begin evac debug
if trial_values['debug']:
    print("evac debug df_experiment")  # Print the entire DataFrame
    print(df_experiment)  # Print the entire DataFrame
    # Print unique animal types
    print("evac debug df_experiment['animal_type'].unique()")  
    print(df_experiment['animal_type'].unique())  # Print unique animal types
    print("evac debug df_trial")  # Print the evacuation data DataFrame
    print(df_trial)  # Print the evacuation data DataFrame
# end evac debug

# Group data by animal type and get evacuation times
# civilians have both speed and vision variable, 4 combinations
# responders have speed and vision fixed, 2 combinations
animal_types = ['able-giraffe', 'able-chameleon', 'challenged-giraffe', 'challenged-chameleon', 'fireman-eagle', 'medical-owl']
evacuation_times = [df_experiment[df_experiment['animal_type'] == animal_type]['evacuation_time'] for animal_type in animal_types]

plt.boxplot(evacuation_times)
# Use animal_types for x-axis labels
# rotate to avoid overwrite
plt.xticks([1, 2, 3, 4, 5, 6], animal_types, rotation=45)  
plt.ylabel('Evacuation Time')
plt.title('Experiment Distribution of Evacuation Times by Person Type')
# write to file
box_file_name = "20250115l-box.png"
# Make sure we call plt.savefig() before calling plt.show().
# plt.show() displays the figure, and in some cases,
# it can clear the figure afterwards. 
plt.savefig(box_file_name)
zip_file.write(box_file_name)

# Generate dual bar chart
summary = decision_tree_results_experiment.groupby('animal_type').agg('sum') 
bar_chart_types = summary.index
matches = summary['match']
misses = summary['missed']
# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
index = range(len(bar_chart_types))

fig, ax = plt.subplots()
rects1 = ax.bar(index, matches, bar_width, label='Match')
# Shift the 'missed' bars slightly to the right
rects2 = ax.bar([i + bar_width for i in index], misses, bar_width, label='Missed')

# Set the labels, title, and legend
ax.set_xlabel('Civilian Type')
ax.set_ylabel('Count')
ax.set_title("Experiment Decision Tree Results")
bar_file_name = "20250115l-bar.png"
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(bar_chart_types, rotation=45)  # Rotate labels if needed
ax.legend()

# Display the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig(bar_file_name)
zip_file.write(bar_file_name)
# display to cell
# matplotlib: With Interactive Mode On: Plots are displayed
# as soon as they are created
# we want this for the experiment boxplot
plt.ion()  # Turn interactive mode on
plt.show()
zip_file.write(file_debug_name)
zip_file.close()
