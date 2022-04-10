import time


def main(params):
    if params['glass_text'] == 'glass':
        if 6 > params['eye_ratio'] > 4:
            if params['eye_close_flag'] == False:
                print("FIRST ROUND STARTED")
                params['eye_close_flag'] = True
                params['eye_close_time_stamp'] = time.time()
            return params
    else:
        if 6 > params['eye_ratio'] > 4.5:
            if params['eye_close_flag'] == False:
                params['eye_close_flag'] = True
                params['eye_close_time_stamp'] = time.time()
            return params

    if params['eye_close_flag'] == True:
        print("FIRST ROUND CLOSED")
        print(f"time : {time.time() - params['eye_close_time_stamp']}")
        params['eye_close_flag'] = False
        params['eye_time_stamp'].append(params['eye_close_time_stamp'])
        params['eye_time_stamp'].append(time.time())
    return params
