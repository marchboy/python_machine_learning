
import time
import random
from datetime import datetime, date, timedelta
from appium import webdriver
from dingtalk_settings import *
from xml.etree.ElementTree import parse

# Date: Oct 14, 2019
# Author: Somebody
# Function: Secret, something unspeakable.


class DingTalk():
    def __init__(self, *args, **kwargs):
        self.driver = None
        self.desired_caps = DESIRED_CAPS
        self.user = LOGIN_INFO.get('user')
        self.passwd = LOGIN_INFO.get('passwd')

    def init_driver(self):
        if self.driver is not None:
            self.driver.quit()

        logs = Log('InitialDriver')
        try:
            self.driver = webdriver.Remote('http://localhost:4723/wd/hub', self.desired_caps)
            logs.info({'errcode':0, 'info':'initial driver successful.'})
        except BaseException as e:
            logs.error({'errcode':100, 'info':'initial driver failed. \n {}'.format(e)})
        
        time.sleep(10)
    
    def skip_sth(self):
        # There maybe some agreement, you need to skip it
        # mostly. you can pass it, beacause it happens occasionally......
        pass

    def client_login(self):
        logs = Log('LogginClient')
        try:
            self.driver.find_element_by_id('com.alibaba.android.rimet:id/et_phone_input').send_keys(self.user)
            self.driver.find_element_by_id('com.alibaba.android.rimet:id/et_pwd_login').send_keys(self.passwd)
            self.driver.find_element_by_id('com.alibaba.android.rimet:id/tv').click()
            logs.info({'errorcode':0, 'info':'login Successful, manually.'})
        except BaseException as e:
            if self.driver.find_element_by_xpath('//android.widget.RelativeLayout[@content-desc="工作"]'):
                logs.info({'errcode':0, 'info':'login successful, naturally.'})
            else:
                # self.skip_sth()
                # pass
                logs.error({'errcode':101, 'info':'login failed. \n {}'.format(e)})     
        else:
            # Not very robust, but enough.
            logs.error({'errcode':102, 'info':'Oops, loging Failed, please check...'})
        finally:
            time.sleep(8)

    def ding_clock(self):
        today = date.today()
        afternoon = datetime(today.year, today.month, today.day) + timedelta(hours=18, minutes=35,seconds=0)
        morning = datetime(today.year, today.month, today.day) + timedelta(hours=9, minutes=36,seconds=0)
        logs = Log('DingClock')
        try:
            self.driver.find_element_by_xpath('//android.widget.RelativeLayout[@content-desc="工作"]').click()
            logs.info({'errcode':0, 'info':'ding work successful, manually'})
            time.sleep(random.randint(6,10))

            self.driver.find_element_by_xpath("/hierarchy/android.widget.FrameLayout/android.widget.Linear" \
                "Layout/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.widget.Linear" \
                "Layout/android.widget.FrameLayout[1]/android.widget.FrameLayout/android.widget.LinearLayout/android.view.View" \
                "Group/android.widget.FrameLayout/android.widget.FrameLayout/android.widget.FrameLayout/android.widget.Frame" \
                "Layout/android.widget.RelativeLayout/android.widget.RelativeLayout/android.widget.FrameLayout/com.uc.webview.export.Web" \
                "View/com.uc.webkit.be/android.webkit.WebView/android.view.View/android.view.View[3]/android.view.View[2]/android.view.View[2]/android.view.View").click()
            logs.info({'errcode':0, 'info':'ding kaoqing successful, manually'})
            time.sleep(random.randint(12,15))
            
            self.driver.find_element_by_xpath('//android.widget.Button[@content-desc="打卡"]').click()
            logs.info({'errcode':0, 'info':'ding daka successful, manually'})
            time.sleep(random.randint(6,10))
            
            if run_time > afternoon:
                try:
                    self.driver.find_element_by_xpath('//android.view.View[@content-desc="下班打卡"]').click()
                    logs.info({'errcode':0, 'info':'dingding successful, manually'})
                except:
                    logs.info({'errcode':0, 'info':'dingding successful, naturally'})

            elif run_time < morning:
                try:
                    self.driver.find_element_by_xpath('//android.view.View[@content-desc="上班打卡"]').click()
                    # send_mail('DingLa, manually')
                    logs.info({'errcode':0, 'info':'dingding successful, manually'})
                except:
                    # send_mail('DingLa, naturally')
                    logs.info({'errcode':0, 'info':'dingding successful, naturally'})
            else:
                logs.info({'errcode':0, 'info':'working time....'})
            time.sleep(random.randint(6,12))
        except BaseException as e:
            # send_mail('Oops,There is something wrong, {}'.format(e))
            logs.error({'errcode':110, 'info':'Oops, clocking Failed, please check. \n {}'.format(e)})

def run():
    ding_talk = DingTalk()
    ding_talk.init_driver()
    
    print('Logging.......')
    ding_talk.client_login()
    print('Clocking......')
    ding_talk.ding_clock()

    ding_talk.driver.quit()

if __name__ == '__main__':
    run_time = datetime.now()
    run()
