from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SESSION_SECRET: str = "change-me"
    SPOTIFY_CLIENT_ID: str
    SPOTIFY_CLIENT_SECRET: str
    SPOTIFY_REDIRECT_URI: str = "http://localhost:8000/api/auth/spotify/callback"
    SPOTIFY_SCOPES: str = "user-read-email"  # add more as needed

    class Config:
        env_file = ".env"

settings = Settings()
