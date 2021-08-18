import requests
from lxml import etree
from datetime import datetime
import urllib.parse as parse
import re
from tqdm import tqdm
import json
import os

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

try:
    month_page = requests.get(CME_month_list_url)
except Exception:
    print('{} requests fail'.format(CME_month_list_url))
html = etree.HTML(month_page.text)
CME_datetime = []
# criteron = "{}/{}/{}".format(year, month, day)
# 查找记录每一次CME的行的tr节点
month_tr_node_list = html.xpath('//tr//td[@headers="hd_date_time"]/..')
# 该列表包含了每一个月的CME发生日期、时间以及标注
pbar = tqdm(total=len(month_tr_node_list))
CME_month_appear_datetime_list = []
for node in month_tr_node_list:
    # datetimelist = node.xpath('/td[@headers="hd_date_time"]/@headers')
    # 获得每一次CME事件的remarks
    remarks = node.xpath('./td[@headers="hd_remark"]/text()')[0].strip()
    # print(remarks)

    start_end_time_partial_url = node.xpath(
        './td[@headers="hd_date_time"][1]/a/@href')[0]  # 路径前若不加点，则代表向该节点的绝对路径，会出错
    start_end_time_url = '/'.join(CME_month_list_url.split('/')
                                  [0:-1])+'/'+start_end_time_partial_url   # start_end_time_url储存了指向每一次CME事件java movie的链接，希望从中获取起止时刻
    # print('/'.join(CME_month_list_url.split('/')[0:-1]))
    # print(start_end_time_partial_url)
    # print(parse.urljoin('/'.join(CME_month_list_url.split('/')
    #       [0:-1]), start_end_time_partial_url))
    # print(parse.urljoin('/'.join(CME_month_list_url.split('/')
    #       [0:-1]), start_end_time_partial_url))
    try:
        CME_java_movie_page = requests.get(start_end_time_url)
    except Exception:
        print('\n{} request fail, jump to next CME incident'.format(
            start_end_time_url))
        print(Exception)
        continue
    time_regex = re.compile(
        r'.*stime=(\d{8})_(\d{4})&etime=(\d{8})_(\d{4})')  # 匹配起止时间
    result = time_regex.findall(CME_java_movie_page.text)[0]
    # 每次CME事件的开始和结束时间，该时间来源于每次CME的java movie
    CME_start_time = datetime.strptime(result[0]+result[1], r'%Y%m%d%H%M')
    CME_end_time = datetime.strptime(result[2]+result[3], r'%Y%m%d%H%M')
    # print(CME_start_time)
    # print(CME_end_time)

    # 每次CME的发生时间和第一次出现的时刻
    datetime_list = node.xpath('./td[@headers="hd_date_time"]/a/text()')
    CME_month_appear_datetime = datetime.strptime(datetime_list[0].strip()+' ' +
                                                  datetime_list[1].strip(), r'%Y/%m/%d %H:%M:%S')
    # print(CME_month_appear_datetime)

    CME_datetime_dict = {'appear': CME_month_appear_datetime, 'start': CME_start_time,
                         'end': CME_end_time, 'remark': remarks}  # 记录每次CME事件日期、起始、结束、标记的字典
    CME_month_appear_datetime_list.append(CME_datetime_dict)
    pbar.update(1)
pbar.close()
# print(CME_month_appear_datetime_list)
# 将包含有每月CME事件的列表储存在该json文件中
if not os.path.exists(os.path.join(save_location, 'CMEList')):
    os.makedirs(os.path.join(save_location, 'CMEList'))
json_filename = os.path.join(
    save_location, 'CMElist\{}_{}_{}_CMEList.json'.format(year, month, day))


# 由于json无法序列化datetime类，所以需要添加此类
class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y%m%d %H%M%S')
        else:
            return json.JSONEncoder.default(self, obj)


with open(json_filename, 'w') as f:
    json.dump(CME_month_appear_datetime_list, f, cls=DataEncoder)
