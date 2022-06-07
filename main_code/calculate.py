import time
from datetime import datetime
import utils as ut

def process_eye_blink_counter(eye_blink_stamp):
    new_eye_blink_stamp = []
    current_time = time.time()
    eye_blink_counter = 0
    result = ""
    if len(eye_blink_stamp) != 0:
        # first check for one hour
        if current_time - eye_blink_stamp[-1] < 1 * 3600:
            new_eye_blink_stamp.append(eye_blink_stamp[-1])
            for i in range(len(eye_blink_stamp) - 1, 0, -1):
                # first check for half hour
                if eye_blink_stamp[i] - eye_blink_stamp[i - 1] < 0.5 * 3600:
                    new_eye_blink_stamp.append(eye_blink_stamp[i - 1])
                    eye_blink_counter = eye_blink_counter + 1
    if eye_blink_counter > 5:
        result = "drowsiness"
        new_eye_blink_stamp = []
    return result, new_eye_blink_stamp


def eye_calculate_main(params):
    params['result_eye'] = ""

    # For eye proessing of blinking
    if len(params['eye_time_stamp']) == 2:
        # print(f"length of eye_time_stamp : {len(eye_time_stamp)}")
        # print(f"time in calculation : {eye_time_stamp[-1] - eye_time_stamp[0]}")
        if params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 30:
            print("EYE CLOSE FOR GREATER THAN 30 SECONDS")
            params["last_blink_trigger_time"] = ut.get_current_time()
            params["last_critical_blink_duration"] = str(params['eye_time_stamp'][-1] - params['eye_time_stamp'][0])[:4]+" seconds"
            params['result_eye'] = "sleeping"
        elif params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 5:
            print("EYE CLOSE FOR GREATER THAN 5 SECONDS")
            # print(eye_time_stamp)
            params["last_blink_trigger_time"] = ut.get_current_time()
            params["last_critical_blink_duration"] = str(params['eye_time_stamp'][-1] - params['eye_time_stamp'][0])[:4]+" seconds"
            params['result_eye'] = "sleeping"

        elif params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 1:
            print("EYE CLOSE FOR GREATER THAN 1 SECOND")
            params["last_blink_trigger_time"] = ut.get_current_time()
            params['eye_blink_stamp'].append(params['eye_time_stamp'][-1])
            params["last_critical_blink_duration"] = str(params['eye_time_stamp'][-1] - params['eye_time_stamp'][0])[:4]+" seconds"
            print(sorted(params['eye_blink_stamp']))

    params['eye_time_stamp'] = []
    if params['result_eye'] == "":
        params['result_eye'], params['eye_blink_stamp'] = process_eye_blink_counter(sorted(params['eye_blink_stamp']))
    return params


def process_mouth_yawn_counter(mouth_yawn_stamp):
    new_mouth_yawn_stamp = []
    current_time = time.time()
    mouth_yawn_counter = 0
    result = ""
    if len(mouth_yawn_stamp) != 0:
        # first check for one hour
        if current_time - mouth_yawn_stamp[-1] < 1 * 3600:
            new_mouth_yawn_stamp.append(mouth_yawn_stamp[-1])
            for i in range(len(mouth_yawn_stamp) - 1, 0, -1):
                # first check for half hour
                if mouth_yawn_stamp[i] - mouth_yawn_stamp[i - 1] < 0.5 * 3600:
                    new_mouth_yawn_stamp.append(mouth_yawn_stamp[i - 1])
                    mouth_yawn_counter = mouth_yawn_counter + 1
    if mouth_yawn_counter > 2:
        result = "drowsiness"
        new_mouth_yawn_stamp = []
    return result, new_mouth_yawn_stamp


def mouth_calculate_main(params):
    params['result_mouth'] = ""
    # For eye proessing of blinking
    if len(params['mouth_time_stamp']) == 2:
        if params['mouth_time_stamp'][-1] - params['mouth_time_stamp'][0] > 1:
            print("MOUTH OPEN FOR GREATER THAN 1 SECOND")
            params['mouth_yawn_stamp'].append(params['mouth_time_stamp'][-1])
            print(sorted(params['mouth_yawn_stamp']))
            params["last_yawn_trigger_time"] = ut.get_current_time()
            params["last_critical_yawn_duration"] = str(params['mouth_time_stamp'][-1] - params['mouth_time_stamp'][0])[:4] + " seconds"

    params['mouth_time_stamp'] = []
    if params['result_mouth'] == "":
        params['result_mouth'], params['mouth_yawn_stamp'] = process_mouth_yawn_counter(
            sorted(params['mouth_yawn_stamp']))
    return params


def head_calculate_main(params):
    params['result_head'] = ""
    # For eye proessing of blinking
    if len(params['head_time_stamp']) == 2:
        if params['head_time_stamp'][-1] - params['head_time_stamp'][0] > 8:
            print("HEAD DOWN FOR GREATER THAN 8 SECOND")
            params['result_head'] = "sleeping"
            params["list_head_time_stamp"].append(params['head_time_stamp'][-1])
            params["last_head_down_trigger_time"] = ut.get_current_time()
            params["last_critical_head_down_duration"] = str(params['head_time_stamp'][-1] - params['head_time_stamp'][0])[:4] + " seconds"
    params['head_time_stamp'] = []
    return params
