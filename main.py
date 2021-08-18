import download_CME_pic as downloadCME
import get_CME_data as getCME

year, month, day = 2013, 8, 16
save_location = r'D:\Programming\CME_data'
CME_month_appear_datetime_list = getCME.get_CME_month_datetime_list(
    year, month)
downloadCME.download_daily_pic(
    year, month, day, save_location, CME_month_appear_datetime_list)
