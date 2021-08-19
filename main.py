import calendar
import download_CME_pic as downloadCME
import get_CME_data as getCME


def download_CME_data_at(start, end, save_location):
    """
    用来下载指定起止时间内的CME数据（起止的年份必须相同）
    Arguments
    ---------
    -start是一个包含起始年、月的元组
    -end是一个包含终止年、月的元组
    -------

    """
    assert start[0] == end[0], '起止和终止的年份必须相同'
    year_month_list = [(start[0], month)
                       for month in range(start[1], end[1]+1)]
    for year, month in year_month_list:
        CME_month_appear_datetime_list = getCME.get_CME_month_datetime_list(
            year, month)
        _, monthlen = calendar.monthrange(year, month)
        for day in range(1, monthlen+1):
            print('download {0:}/{1:0>2d}/{2:0>2d}'.format(year, month, day))
            downloadCME.download_daily_pic(
                year, month, day, save_location, CME_month_appear_datetime_list)


if __name__ == '__main__':
    start = (2013, 8)
    end = (2013, 12)
    save_location = r'D:\Programming\CME_data'
    download_CME_data_at(start, end, save_location=save_location)
