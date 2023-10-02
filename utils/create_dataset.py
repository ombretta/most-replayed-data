import requests as req
import json
import pandas as pd
import datetime

YT_OPERATIONAL_API_URL = "http://localhost/YouTube-operational-API/" # "https://yt.lemnoslife.com"

# https://www.youtube.com/results?search_query=vlog&sp=EgQYAzAB
#r = req.get(YT_OPERATIONAL_API_URL+'/videos?part=mostReplayed&id={}'.format(video_id))

VIDEO_COUNT_TARGET = 10 # 500000
MAX_RESULTS = 50 # per GET operation
videos = []
DAYS_AGO = 365*3
base = datetime.datetime.today() - datetime.timedelta(days=600)
date_ranges = [((base - datetime.timedelta(days=x)).isoformat("T", timespec='seconds') + "Z", (base - datetime.timedelta(days=x-7)).isoformat("T", timespec='seconds') + "Z") for x in range(0,DAYS_AGO,7)]

print(date_ranges)
# for each week
for published_after, published_before in date_ranges:
    next_page_token = None
    # pagination loop
    while True:
        if next_page_token:
            next_page_token_param = f"pageToken={next_page_token}"
        else:
            next_page_token_param = ""
        req_str = YT_OPERATIONAL_API_URL+'/search?part=id,snippet&q=vlog&type=video&videoDuration=medium&videoLicense=creativeCommon'+'&publishedBefore='+published_before+'&maxResults='+ \
            str(MAX_RESULTS)+'&'+next_page_token_param
        print(req_str)
        r = req.get(req_str)
        json_response = json.loads(r.text)
        videos.extend([{
            "id": d["id"]["videoId"],
            "title": d["snippet"]["title"],
            "duration": d["snippet"]["duration"],
            "views": d["snippet"]["duration"],
            "channelId": d["snippet"]["channelId"], 
            "channelTitle": d["snippet"]["channelTitle"],
            "timestamp": d["snippet"]["timestamp"]
            }
            for d in json_response["items"] if d["id"]["videoId"] 
            ])
        # break when there is no next page
        if not "nextPageToken" in json_response.keys():
            break
        else:
            next_page_token = json_response["nextPageToken"]
    print(len(videos))
    if(len(videos) > VIDEO_COUNT_TARGET):
        break
print(len(videos))
videos_df = pd.DataFrame(videos)
videos_df.to_csv("./vlog_250k_dataset_02.csv")