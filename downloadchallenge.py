from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(path)
downloader.downloadGames(files=[f"1_224p.mkv", f"2_224p.mkv"], split=['challenge'], verbose=False, randomized=True)
