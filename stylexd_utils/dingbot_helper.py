#!/usr/bin/python3
#!--*-- coding:utf-8 --*--
import os
import json
import time
import hmac
import hashlib
import urllib
import base64
import requests
import urllib.parse

_PROFILE_FILE = os.path.join(os.path.dirname(__file__), "..", "info.json")


def ding_msg(msg):
    """
    向钉钉机器人发送需要发送的消息

    param: msg 需要发送的内容(比如下面的markdown)
    return: 钉钉机器人的响应
    """

    data = json.load(open(_PROFILE_FILE, "rb"))
    token, secret = data["dingbot"]["token"], data["dingbot"]["secret"]

    webhook = f"https://oapi.dingtalk.com/robot/send?access_token={token}"

    data = json.dumps(msg)

    start = time.time()
    timestamp = int(round(start * 1000))
    secret_enc = secret.encode("utf-8")
    string_to_sign = "{}\n{}".format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode("utf-8")
    hmac_code = hmac.new(
        secret_enc, string_to_sign_enc, digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    sign_webhook = f"{webhook}&timestamp={timestamp}&sign={sign}"

    headers = {"Content-Type": "application/json;charset=utf-8"}

    req = requests.post(url=sign_webhook, headers=headers, data=data)

    return req


"""
消息类型: https://open.dingtalk.com/document/group/message-types-and-data-format
https://oapi.dingtalk.com/robot/send?access_token=f01774c2790afbccc13be337948d5ff7f08c91ad117cd3e31315c92b4c536e1b
"""


def markdown_msg(title, markdown_text, at=""):
    """
    将字符串转换成markdown格式, 用于钉钉机器人发送

    param: title 消息标题
    param: markdown_text 消息征文
    return:
    """
    msg = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": markdown_text},
        "at": {
            "atMobiles": [
                at,
            ],
            "isAtAll": not at,
        },
    }

    return msg


def text_msg(content, at=""):
    """
    将字符串转换成text格式, 用于钉钉机器人发送

    param: title 消息标题
    param: text 消息征文
    return:
    """
    msg = {
        "msgtype": "text",
        "text": {"content": content},
        "at": {
            "atMobiles": [
                at,
            ],
            # @ 所有人
            "isAtAll": False,
        },
    }

    return msg


def send_ding_msg(title, markdown_text):
    msg = markdown_msg(title, markdown_text)
    ding_msg(msg)


if __name__ == "__main__":
    msg = markdown_msg("本地监控", "this is a test msg.")
    res = ding_msg(msg)

    print("[INFO] Done.", res)
