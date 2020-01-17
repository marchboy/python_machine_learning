
import time
from appium import webdriver
from dingtalk_settings import *




class DingTalk():
    def __init__(self, *args, **kwargs):
        self.driver = None
        self.desired_caps = DESIRED_CAPS

    def init_driver(self):
        if self.driver is not None:
            self.driver.quit()
        self.driver = webdriver.Remote('http://localhost:4723/wd/hub', self.desired_caps)
        time.sleep(10)
        # self.driver.find_element_by_id('com.alibaba.android.rimet:id/tv').click()

def main():
    process = DingTalk()
    process.init_driver()
    time.sleep(10)
    process.driver.quit()

if __name__ == '__main__':
    main()

    print(LOGIN_INFO)
    print(DESIRED_CAPS)

