from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader('/content/soccetnet')
downloader.password = 's0cc3rn3t'
downloader.downloadGames(files=[f"1_224p.mkv", f"2_224p.mkv"], split=['challenge'], verbose=False, randomized=True)
