import requests
from datetime import datetime
import re
from tqdm import tqdm
import json
import os
import urllib.parse as parse
import threading

save_location = r'D:\Programming\CME_data'
year, month, day = 2013, 8, 1
# 记录所有CME的链接
CME_all_list_url = 'https://cdaw.gsfc.nasa.gov/CME_list/'
# 记录每月CME的链接
CME_month_list_url = 'https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/{0}_{1:0>2d}/univ{0}_{1:0>2d}.html'.format(
    year, month)
# 记录每日CME的图像的链接
CME_daily_pics_url = 'https://cdaw.gsfc.nasa.gov/images/soho/lasco/{0:}/{1:0>2d}/{2:0>2d}/'.format(
    year, month, day)
json_filename = os.path.join(
    save_location, 'CMElist\{}_{}_{}_CMEList.json'.format(year, month, day))

# with open(json_filename, 'r') as f:
#     CME_month_appear_datetime_list = json.load(f)
# for CME_month_appear_datetime in CME_month_appear_datetime_list:
#     CME_month_appear_datetime['appear'] = datetime.strptime(
#         CME_month_appear_datetime['appear'], '%y%m%d %H%M%S')
#     CME_month_appear_datetime['start'] = datetime.strptime(
#         CME_month_appear_datetime['start'], '%y%m%d %H%M%S')
#     CME_month_appear_datetime['end'] = datetime.strptime(
#         CME_month_appear_datetime['end'], '%y%m%d %H%M%S')


def read_CME_list_json(json_filename):
    # 从CME json文件夹中还原数据，重新生成一个包含一个月中CME事件的列表
    with open(json_filename, 'r') as f:
        CME_month_appear_datetime_list = json.load(f)
    for CME_month_appear_datetime in CME_month_appear_datetime_list:
        CME_month_appear_datetime['appear'] = datetime.strptime(
            CME_month_appear_datetime['appear'], '%Y%m%d %H%M%S')
        CME_month_appear_datetime['start'] = datetime.strptime(
            CME_month_appear_datetime['start'], '%Y%m%d %H%M%S')
        CME_month_appear_datetime['end'] = datetime.strptime(
            CME_month_appear_datetime['end'], '%Y%m%d %H%M%S')
    return CME_month_appear_datetime_list


def determine_CME(pic_time: datetime, CME_month_appear_datetime_list: list):
    remark = None
    is_CME = False
    for CME_incident in CME_month_appear_datetime_list:
        if pic_time > CME_incident['appear'] and pic_time < CME_incident['end']:
            is_CME = True
            remark = CME_incident['remark']
            # 发现该图片在哪一次CME事件期间，就可以停止继续遍历后面的事件
            break
    # 若图片的pictime不在任何一次CME事件的起止时间内，认为该图片不是CME，其remark和is_CME就是None和False
    return remark, is_CME


CME_month_appear_datetime_list = read_CME_list_json(json_filename)
print(CME_month_appear_datetime_list)
daily_pic_list = []  # 包含该日所有图片的链接、remarks以及是否是CME事件
daily_pic_page = requests.get(CME_daily_pics_url)
pic_href_regex = re.compile(r'<a href="\d{8}_\d{6}_lasc2rdf_aia193rdf.png">')
hrefs = pic_href_regex.findall(daily_pic_page.text)
pic_datetime_regex = re.compile(
    'href="(\d{8})_(\d{6})_')  # 用来识别href中图片时间的正则表达式
pic_href_regex = re.compile('href="(.*)"')  # 用来识别href中图片文件名的正则表达式
for href in hrefs:
    # pic_time包含了该图片的拍摄时间
    pic_time = pic_datetime_regex.findall(
        href)[0][0]+' '+pic_datetime_regex.findall(href)[0][1]
    pic_time = datetime.strptime(pic_time, '%Y%m%d %H%M%S')

    pic_dict = {}  # 包含了该图片url、标记、是否是CME的字典
    # 一张图片的URL类似于
    # https://cdaw.gsfc.nasa.gov/images/soho/lasco/2013/08/01/20130801_004805_lasc2rdf.png
    pic_filename = pic_href_regex.findall(href)[0]
    url = parse.urljoin(CME_daily_pics_url, pic_filename)
    pic_dict['url'] = url
    # is_CME = False
    # pic_dict['remark'] = None
    # pic_dict['is_CME'] = is_CME
    remark, is_CME = determine_CME(pic_time, CME_month_appear_datetime_list)
    pic_dict['remark'] = remark
    pic_dict['is_CME'] = is_CME
    # for CME_incident in CME_month_appear_datetime_list:
    #     if pic_time > CME_incident['appear'] and pic_time < CME_incident['end']:
    #         is_CME = True
    #         remark = CME_incident['remark']
    #         pic_dict['remark'] = remark
    #         pic_dict['is_CME'] = is_CME
    #         break  # 发现该图片在哪一次CME事件期间，就可以停止继续遍历后面的事件
    daily_pic_list.append(pic_dict)
for daily_pic in daily_pic_list:
    print(daily_pic)
