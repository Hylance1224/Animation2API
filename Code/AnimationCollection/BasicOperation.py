import re
import os
import time


# # 获取当前的app以及当前的activity
# def get_package_and_activity(global_d, target_package_name):
#     out = os.popen('adb shell dumpsys window windows | findstr mFocusedApp').read()
#     # print(out)
#     if 'null' in out:
#         global_d.press("home")
#         time.sleep(2)
#         global_d.app_start(target_package_name)
#         out = os.popen('adb shell dumpsys window windows | findstr mFocusedApp').read()
#     pattern = re.compile(r"\{[^{}]+\}")
#     list = pattern.findall(out)
#     s = list[0]
#     num = s.find('u0 ')
#     num1 = s.rfind('/')
#     num2 = s.rfind(' ')
#     package_name = s[num+3: num1]
#     activity = s[num1 + 1:num2]
#     return package_name, activity

# 获取当前的app以及当前的activity
def get_package_and_activity(global_d, target_package_name):
    package_activity = global_d.app_current()
    package_name = package_activity['package']
    activity = package_activity['activity']
    return package_name, activity


def stop_app(global_d):
    running_apps = global_d.app_list_running()
    for i in running_apps:
        if i != 'com.github.shadowsocks' and i != 'com.github.uiautomator' and i != 'com.android.systemui' \
                and i != 'com.android.vending' and i != 'com.google.android.gms':
            # print(i)
            global_d.app_stop(i)


def stop_other_app(global_d, package_name):
    running_apps = global_d.app_list_running()
    for i in running_apps:
        # print(i)
        if i != 'com.github.shadowsocks' and i != package_name and i != 'com.github.uiautomator' \
                and i != 'com.android.systemui' and i != 'com.android.vending' and i != 'com.google.android.gms':
            # print(i)
            global_d.app_stop(i)
            # time.sleep(10)


