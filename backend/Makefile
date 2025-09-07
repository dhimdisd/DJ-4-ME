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

## Auto-detect (playlist or single) from ID or URL
## Examples:
##   make download INPUT=PLEguQ1EpXp3Tb1g6xosTsBRVmAV-fBq6d          # playlist ID
##   make download INPUT=WyM4pZC1h1o                                  # video ID
##   make download INPUT="https://music.youtube.com/playlist?list=..." # playlist URL
##   make download INPUT="https://www.youtube.com/watch?v=..."         # video URL
download:
	cd $(BACKEND_DIR) && \
	python dj_cli.py download "$(INPUT)" \
	  --site $(TRACK_SITE) \
	  --outdir downloads/audio \
	  --codec $(CODEC) \
	  --mp3-bitrate $(MP3_BR) \
	  $(DL_FLAGS)

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


AUDIO_DIR := v2_yt_download/downloads/audio

# Delete a single track by TITLE or by PATH
# Usage:
#   make delete-track TITLE="No Broke Boys"
#   make delete-track PATH="v2_yt_download/downloads/audio/No Broke Boys.mp3"
delete-track:
	@if [ -n "$(TITLE)" ]; then \
	  python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) delete-track "$$TITLE"; \
	elif [ -n "$(PATH)" ]; then \
	  python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) delete-track "$$PATH"; \
	else \
	  echo "Usage: make delete-track TITLE='Exact Title'  OR  make delete-track PATH='path/to/file.mp3'"; \
	  exit 1; \
	fi

# Delete an entire playlist by its title (as recorded in ledger)
# Usage:
#   make delete-playlist TITLE="Afro House 2025"
delete-playlist:
	@if [ -z "$(TITLE)" ]; then \
	  echo "Usage: make delete-playlist TITLE='Playlist Title'"; exit 1; \
	fi
	@python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) delete-playlist "$(TITLE)"

# Dry-run wrappers (preview only)
delete-track-dry:
	@if [ -n "$(TITLE)" ]; then \
	  python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) --dry-run delete-track "$$TITLE"; \
	elif [ -n "$(PATH)" ]; then \
	  python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) --dry-run delete-track "$$PATH"; \
	else \
	  echo "Usage: make delete-track-dry TITLE='Exact Title'  OR  make delete-track-dry PATH='path/to/file.mp3'"; \
	  exit 1; \
	fi

delete-playlist-dry:
	@if [ -z "$(TITLE)" ]; then \
	  echo "Usage: make delete-playlist-dry TITLE='Playlist Title'"; exit 1; \
	fi
	@python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) --dry-run delete-playlist "$(TITLE)"

# Clean orphans: broken symlinks + ledger rows pointing to missing files
clean-orphans:
	@python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) clean-orphans


# Delete only playlist symlinks + ledger/archive rows; keep audio files
unlink-playlist:
	@if [ -z "$(TITLE)" ]; then \
	  echo "Usage: make unlink-playlist TITLE='Playlist Title'"; exit 1; \
	fi
	@python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) --keep-audio delete-playlist "$(TITLE)"

unlink-playlist-dry:
	@if [ -z "$(TITLE)" ]; then \
	  echo "Usage: make unlink-playlist-dry TITLE='Playlist Title'"; exit 1; \
	fi
	@python v2_yt_download/util/library_admin.py --outdir $(AUDIO_DIR) --keep-audio --dry-run delete-playlist "$(TITLE)"