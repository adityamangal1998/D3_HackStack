import time

def eye_without_glass(params):
    if params['head_text_1'] == "Forward":
        if params['eye_ratio'] > 5.0:
            return True
    elif params['head_text_1'] == "Looking Right":
        if params['eye_ratio'] > 4.7:
            return True
    elif params['head_text_1'] == "Looking Left":
        if params['eye_ratio'] > 4.5:
            return True
    elif params['head_text_1'] == "Looking Up":
        if params['eye_ratio'] > 7.0:
            return True
    elif params['head_text_1'] == "Looking Down":
        if params['eye_ratio'] > 4.0:
            return True
    return False

def eye_with_glass(params):
    if params['head_text_1'] == "Forward":
        if params['eye_ratio'] > 3.8:
            return True
    elif params['head_text_1'] == "Looking Right":
        if params['eye_ratio'] > 3.8:
            return True
    elif params['head_text_1'] == "Looking Left":
        if params['eye_ratio'] > 3.8:
            return True
    elif params['head_text_1'] == "Looking Up":
        if params['eye_ratio'] > 3.8:
            return True
    elif params['head_text_1'] == "Looking Down":
        if params['eye_ratio'] > 3.8:
            return True
    return False

def eye_main(params):
    if params['glass_text'] == 'glass':
        if eye_with_glass(params):
            if params['eye_close_flag'] == False:
                print("FIRST ROUND STARTED")
                params['eye_close_flag'] = True
                params['eye_close_time_stamp'] = time.time()
            else:
                if time.time() - params['eye_close_time_stamp'] > 5:
                    print("FIRST ROUND CLOSED OF EYE")
                    params['eye_close_flag'] = False
                    params['eye_time_stamp'].append(params['eye_close_time_stamp'])
                    params['eye_time_stamp'].append(time.time())
            return params
    else:
        if eye_without_glass(params):
            if params['eye_close_flag'] == False:
                print("FIRST ROUND STARTED")
                params['eye_close_flag'] = True
                params['eye_close_time_stamp'] = time.time()
            else:
                if time.time() - params['eye_close_time_stamp'] > 5:
                    print("FIRST ROUND CLOSED OF EYE")
                    params['eye_close_flag'] = False
                    params['eye_time_stamp'].append(params['eye_close_time_stamp'])
                    params['eye_time_stamp'].append(time.time())
            return params

    if params['eye_close_flag'] == True:
        print("FIRST ROUND CLOSED")
        print(f"time : {time.time() - params['eye_close_time_stamp']}")
        params['eye_close_flag'] = False
        params['eye_time_stamp'].append(params['eye_close_time_stamp'])
        params['eye_time_stamp'].append(time.time())
    return params

def mouth_main(params):
    if params['mouth_ratio'] > 29:
        if params['mouth_open_flag'] == False:
            print("FIRST ROUND STARTED OF MOUTH")
            params['mouth_open_flag'] = True
            params['mouth_open_time_stamp'] = time.time()
        return params
    if params['mouth_open_flag'] == True:
        print("FIRST ROUND CLOSED OF MOUTH")
        print(f"time : {time.time() - params['mouth_open_time_stamp']}")
        params['mouth_open_flag'] = False
        params['mouth_time_stamp'].append(params['mouth_open_time_stamp'])
        params['mouth_time_stamp'].append(time.time())
    return params


def head_main(params):
    if params['head_text_1'] == "Looking Down":
        if params['head_down_flag'] == False:
            print("FIRST ROUND STARTED OF HEAD")
            params['head_down_flag'] = True
            params['head_down_time_stamp'] = time.time()
        else:
            if time.time() - params['head_down_time_stamp'] > 8:
                print("FIRST ROUND CLOSED OF HEAD")
                params['head_down_flag'] = False
                params['head_time_stamp'].append(params['head_down_time_stamp'])
                params['head_time_stamp'].append(time.time())
        return params

    if params['head_down_flag'] == True:
        print("FIRST ROUND CLOSED OF HEAD")
        print(f"time : {time.time() - params['head_down_time_stamp']}")
        params['head_down_flag'] = False
        params['head_time_stamp'].append(params['head_down_time_stamp'])
        params['head_time_stamp'].append(time.time())
    return params
