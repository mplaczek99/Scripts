import os
from mutagen.easyid3 import EasyID3
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-modify-private,user-library-modify"))

def find_track(artist, title):
    q = f"track:{title} artist:{artist}"
    results = sp.search(q, type="track", limit=1)
    items = results["tracks"]["items"]
    if items:
        return items[0]["id"]
    return None

playlist_name = "My MP3 Matches"
playlist = sp.user_playlist_create(sp.current_user()["id"], playlist_name, public=False)

music_dir = os.path.expanduser("~/Music")
for file in os.listdir(music_dir):
    if file.endswith(".mp3"):
        path = os.path.join(music_dir, file)
        try:
            tags = EasyID3(path)
            artist = tags.get("artist", [""])[0]
            title = tags.get("title", [""])[0]
            track_id = find_track(artist, title)
            if track_id:
                sp.playlist_add_items(playlist["id"], [track_id])
                print(f"Added: {artist} - {title}")
            else:
                print(f"No match: {artist} - {title}")
        except Exception as e:
            print(f"Error with {file}: {e}")

