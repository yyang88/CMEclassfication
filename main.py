import calendar
import download_CME_pic as downloadCME
import get_CME_data as getCME
import threading


def mutilthread_download_daily_pic(year, month, days, save_location,
                                   CME_month_appear_datetime_list):
    """
    实现多线程下载每月的数据，是对downloadCME.download_daily_pic的包装
    Arguments:
    ---------
    year,month                      = 是给定的年和月
    days                            = 一个列表，包含了这个月的每一日
    save_location                   = 数据存储位置的根目录
    CME_month_appear_datetime_list  = 每月的CME数据
    -------

    """
    while True:
        if days:
            day = days.pop()
            print('download {0:}/{1:0>2d}/{2:0>2d}'.format(year, month, day))
            downloadCME.download_daily_pic(
                year, month, day, save_location, CME_month_appear_datetime_list)
        else:
            break


def download_CME_data_at(start, end, save_location, num_threads):
    """
    用来下载指定起止时间内的CME数据（起止的年份必须相同）
    Arguments
    ---------
    start    = 一个包含起始年、月的元组
    end      = 一个包含终止年、月的元组
    -------

    """
    assert start[0] == end[0], '起止和终止的年份必须相同'
    year_month_list = [(start[0], month)
                       for month in range(start[1], end[1]+1)]
    for year, month in year_month_list:
        CME_month_appear_datetime_list = getCME.get_CME_month_datetime_list(
            year, month)
        _, monthlen = calendar.monthrange(year, month)
        days = list(range(1, monthlen+1))
        thread_pool = [threading.Thread(target=mutilthread_download_daily_pic,
                                        args=(year, month, days, save_location,
                                              CME_month_appear_datetime_list))
                       for i in range(num_threads)]
        for thread in thread_pool:
            thread.start()
        for thread in thread_pool:
            thread.join()


if __name__ == '__main__':
    start = (2013, 8)
    end = (2013, 8)
    save_location = r'D:\Programming\CME_data'
    num_threads = 4
    download_CME_data_at(start, end, save_location, num_threads)
