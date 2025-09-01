# ====== Config (override via: make TARGET VAR=value) ======
BACKEND_DIR      := v2_yt_download
OUTDIR           ?= $(BACKEND_DIR)/downloads/audio
PLAYLISTS_ROOT   ?= $(BACKEND_DIR)/downloads/dj_playlists
METADATA_DIR     ?= $(BACKEND_DIR)/downloads/metadata

CODEC            ?= mp3         # mp3|wav
MP3_BR           ?= 320         # mp3 bitrate
BROWSER          ?= chrome      # for cookies; or set COOKIEFILE
FORCE            ?= 0           # 1 => force redownload (ignore archive + overwrite)
NOARCHIVE        ?= 0           # 1 => ignore archive (no overwrite)

# ====== URL builders (pass ID or full URL) ======
# If you pass PLAYLIST_URL or TRACK_URL directly, those win.
# Else we compose from IDs and the site choice.
PLAYLIST_SITE    ?= music       # music|youtube (default playlists → YouTube Music)
TRACK_SITE       ?= youtube     # music|youtube (default singles → YouTube)

PLAYLIST_URL_YT     = https://www.youtube.com/playlist?list=$(PLAYLIST_ID)
PLAYLIST_URL_MUSIC  = https://music.youtube.com/playlist?list=$(PLAYLIST_ID)
TRACK_URL_YT        = https://www.youtube.com/watch?v=$(TRACK_ID)
TRACK_URL_MUSIC     = https://music.youtube.com/watch?v=$(TRACK_ID)

ifeq ($(PLAYLIST_SITE),music)
  SELECTED_PLAYLIST_URL = $(if $(PLAYLIST_URL),$(PLAYLIST_URL),$(PLAYLIST_URL_MUSIC))
else
  SELECTED_PLAYLIST_URL = $(if $(PLAYLIST_URL),$(PLAYLIST_URL),$(PLAYLIST_URL_YT))
endif

ifeq ($(TRACK_SITE),music)
  SELECTED_TRACK_URL = $(if $(TRACK_URL),$(TRACK_URL),$(TRACK_URL_MUSIC))
else
  SELECTED_TRACK_URL = $(if $(TRACK_URL),$(TRACK_URL),$(TRACK_URL_YT))
endif

# ====== Common flags assembly ======
DL_FLAGS =
ifeq ($(FORCE),1)
  DL_FLAGS += --force
endif
ifeq ($(NOARCHIVE),1)
  DL_FLAGS += --no-archive
endif
ifneq ($(COOKIEFILE),)
  DL_FLAGS += --cookiefile $(COOKIEFILE)
else
  ifneq ($(BROWSER),)
    DL_FLAGS += --browser $(BROWSER)
  endif
endif

# ====== Targets ======

## Download a playlist by ID or URL
## Examples:
##   make download-playlist PLAYLIST_ID=PLEguQ...
##   make download-playlist PLAYLIST_URL="https://music.youtube.com/playlist?list=..."
download-playlist:
	cd $(BACKEND_DIR) && \
	python dj_cli.py download-playlist "$(SELECTED_PLAYLIST_URL)" \
	  --outdir downloads/audio \
	  --codec $(CODEC) \
	  --mp3-bitrate $(MP3_BR) \
	  $(DL_FLAGS)

## Download a single track by ID or URL
## Examples:
##   make download-one TRACK_ID=WyM4pZC1h1o
##   make download-one TRACK_URL="https://www.youtube.com/watch?v=WyM4pZC1h1o"
download-one:
	cd $(BACKEND_DIR) && \
	python dj_cli.py download-one "$(SELECTED_TRACK_URL)" \
	  --outdir downloads/audio \
	  --codec $(CODEC) \
	  --mp3-bitrate $(MP3_BR) \
	  $(DL_FLAGS)

## Force re-download a playlist (ignore archive + overwrite)
## Example:
##   make force-playlist PLAYLIST_ID=PLEguQ...
force-playlist:
	$(MAKE) download-playlist PLAYLIST_ID="$(PLAYLIST_ID)" PLAYLIST_URL="$(PLAYLIST_URL)" PLAYLIST_SITE="$(PLAYLIST_SITE)" FORCE=1

## Analyze every audio file in downloads/audio
analyze-all:
	cd $(BACKEND_DIR) && \
	python dj_cli.py analyze-dir downloads/audio

## Analyze a single file
## Example:
##   make analyze-one FILE="downloads/audio/Some Track.wav"
analyze-one:
	cd $(BACKEND_DIR) && \
	python dj_cli.py analyze-file "$(FILE)"

## Organize audio + metadata into subfolders
organize:
	cd $(BACKEND_DIR) && \
	python organize_files.py

## Clean only playlist symlinks (keeps audio + metadata)
clean-playlists:
	rm -rf $(PLAYLISTS_ROOT)/*

## Nuke audio, metadata, and playlist symlinks (careful!)
clean:
	rm -rf $(OUTDIR)/* $(METADATA_DIR)/* $(PLAYLISTS_ROOT)/*
