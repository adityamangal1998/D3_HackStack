import time


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
    return result, new_eye_blink_stamp



def main(params):
    params['result'] = ""

    # For eye proessing of blinking
    if len(params['eye_time_stamp']) == 2:
        # print(f"length of eye_time_stamp : {len(eye_time_stamp)}")
        # print(f"time in calculation : {eye_time_stamp[-1] - eye_time_stamp[0]}")
        if params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 30:
            print("EYE CLOSE FOR GREATER THAN 30 SECONDS")
            params['result'] = "sleeping"
        elif params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 5:
            print("EYE CLOSE FOR GREATER THAN 5 SECONDS")
            # print(eye_time_stamp)
            params['result'] = "sleeping"

        elif params['eye_time_stamp'][-1] - params['eye_time_stamp'][0] > 1:
            print("EYE CLOSE FOR GREATER THAN 1 SECOND")
            params['eye_blink_stamp'].append(params['eye_time_stamp'][-1])
            print(sorted(params['eye_blink_stamp']))

    params['eye_time_stamp'] = []
    if params['result'] == "":
        params['result'], params['eye_blink_stamp'] = process_eye_blink_counter(sorted(params['eye_blink_stamp']))
    return params
