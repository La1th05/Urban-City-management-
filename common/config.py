from pydantic_settings import SettingsConfigDict, BaseSettings

class Settings(BaseSettings):
    H:int
    W:int
    D:int
    C:int
    T:int
    METERS_PER_PIXEL:float
    LAND_USE:list[str]
    EUROSAT_CLASSES:list[str]
            
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",case_sensitive=False,extra="ignore")
    
def get_settings():
        return Settings()